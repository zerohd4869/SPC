import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import AutoTokenizer, AutoModel

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class KLDivLoss(torch.nn.Module):
    """
    The KL-divergence term in the probabilistic coding objective.
    """

    def __init__(self):
        super(KLDivLoss, self).__init__()

    def forward(self, mu, logvar, label_ids=None, mask_key=0):
        kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2

        if label_ids is not None:
            indices = [k for k, v in enumerate(label_ids) if v == mask_key]
            kl_loss = kl_loss[indices] if len(indices) > 0 else None

        if kl_loss is not None:
            kl_loss = kl_loss.mean(dim=1).mean()

        return kl_loss


class BELoss(torch.nn.Module):
    """
    The batch entropy term in the structured regularization objective.
    """

    def __init__(self):
        super(BELoss, self).__init__()

    def entropy(self, logits):
        return -1.0 * (F.softmax(logits, dim=0) * F.log_softmax(logits, dim=0)).sum()

    def forward(self, logits):
        mean_output = torch.mean(logits, dim=0)
        return -1.0 * self.entropy(mean_output).cuda()


class TaskModule(nn.Module):
    """
    TaskModule: a deterministic head for the downstream task
    """

    def __init__(self, hidden_size, dropout=0.2, output_classes=2):
        super().__init__()
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_classes))

    def forward(self, x):
        output = self.output_layer(x)
        return output


class StochasticTaskModule(nn.Module):
    """
    StochasticTaskModule: a stochastic head for the downstream task
    """

    def __init__(self, input_dim=768, dropout=0.0, output_dim=3):
        super(StochasticTaskModule, self).__init__()
        self.mu_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

        self.logvar_head = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        embedding = self._reparameterize(mu, logvar)
        return (mu, logvar, embedding)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std


class SPC_PLM(nn.Module):
    """
    [AAAI'24 Oral] Structured Probabilistic Coding

    SPC is a supervised representation learning technology, which serves as an encoder-only probabilistic coding framework with structured regularization from the target space.
    In the implementation, we leverage pre-trained language models, such as BERT and RoBERTa, as the backbone models for fine-tuning on downstream tasks.
    """

    def __init__(self, pretrained_model_path="bert-base-chinese", pooling_method="cls",
                 max_length=128, dropout=0.2, var_weight=0.0, clu_weight=0.0,
                 tasks_config=None, output_hidden_states=False, task_type="cls",
                 module_print_flag=False, normalize_flag=False, tokenizer_add_e_flag=False):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        self.bert = AutoModel.from_pretrained(pretrained_model_path)
        if tokenizer_add_e_flag:
            self.tokenizer.add_special_tokens({'additional_special_tokens': ["<e>", "<e/>"]})
            self.bert.resize_token_embeddings(len(self.tokenizer))

        self.hidden_size = self.bert.config.hidden_size
        self.max_length = max_length
        self.tasks_config = tasks_config
        self.pooling_method = pooling_method
        self.output_hidden_states = output_hidden_states
        self.var_weight = var_weight
        self.clu_weight = clu_weight
        self.normalize_flag = normalize_flag
        self.task_type = task_type

        if self.var_weight == 0:
            # deterministic task module
            self.task_module = nn.ModuleDict(
                dict((k.value,
                      TaskModule(hidden_size=self.hidden_size, dropout=dropout, output_classes=v["num_classes"]))
                     for k, v in tasks_config.items())
            )
        else:
            # stochastic task module
            self.var_task_modules = nn.ModuleDict(
                dict((k.value,
                      StochasticTaskModule(input_dim=self.hidden_size, dropout=dropout, output_dim=v["num_classes"]))
                     for k, v in tasks_config.items())
            )

        if task_type == "cls":
            # For single- or multi-task setting for classification
            self.task_criterion = CrossEntropyLoss(weight=None, reduction='mean')
        elif task_type == "res":
            # For single- or multi-task setting for regression
            self.task_criterion = MSELoss(size_average=True)
        elif task_type == "multi":
            # For cross-type multi-task setting
            self.task_criterion = {
                "cls": CrossEntropyLoss(weight=None, reduction='mean'),
                "res": MSELoss(size_average=True)
            }

        self.KLDivLoss = KLDivLoss() if self.var_weight > 0 else None
        self.BELoss = BELoss() if self.clu_weight > 0 else None

        if module_print_flag: print(self)

    def forward(self, x, label, task):
        tokenized_input = self.tokenizer(text=x, text_pair=None, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length)
        for name, data in tokenized_input.items():
            tokenized_input[name] = tokenized_input[name].to(device)
        tokenized_input["output_hidden_states"] = self.output_hidden_states

        outputs = self.bert(**tokenized_input)

        if self.pooling_method == "cls":
            hidden = outputs.last_hidden_state[:, 0, :]
        else:
            hidden = None
            print("pooling_method error !")
            exit(0)

        kl_loss = None
        b_loss = None
        mu = None
        if task in self.tasks_config.keys():
            if self.var_weight > 0:
                if self.normalize_flag:
                    hidden = F.normalize(hidden, dim=-1)

                mu, logvar, x_bias = self.var_task_modules[task.value](hidden)
                pred = torch.zeros_like(mu).type(mu.type())

                if self.training:
                    if self.tasks_config[task]['task_type'] == "res":
                        kl_loss = self.KLDivLoss(mu, logvar, label_ids=None, mask_key=-1)
                        pred = x_bias
                    else:
                        for i in range(self.tasks_config[task]["num_classes"]):
                            kl_loss_ = self.KLDivLoss(mu, logvar, label_ids=label, mask_key=i)
                            if kl_loss_ is not None:
                                kl_loss = kl_loss_ if kl_loss is None else kl_loss + kl_loss_
                            indices = [k for k, v in enumerate(label) if v == i]
                            if len(indices) > 0:
                                pred[indices] = x_bias[indices]
                    if self.clu_weight > 0:
                        b_loss = self.BELoss(mu)
                else:
                    pred = mu

            else:
                pred = self.task_module[task.value](F.relu(hidden))

            task_criterion = self.task_criterion if self.task_type != "multi" else self.task_criterion[self.tasks_config[task]['task_type']]
            if type(label) == list:
                loss = None
                for i in range(len(label)):
                    if i == 0:
                        loss = task_criterion(pred[:, i], label[i])
                    else:
                        loss += task_criterion(pred[:, i], label[i])
                if len(label) > 1: loss = loss / len(label)
            else:
                loss = task_criterion(pred, label)

            if self.var_weight > 0 and kl_loss is not None:
                # PC loss in the paper: MLE + KL
                loss += kl_loss * self.var_weight
                if self.clu_weight > 0:
                    # SPC loss in the paper: MLE + KL + BEL
                    loss += b_loss * self.clu_weight

                print("[MLE+KL+BEL] task_loss: {}, kl_loss: {} * w {}, bel_loss: {} * w {}".format(
                    loss.item(), kl_loss.item(), self.var_weight, b_loss.item() if self.clu_weight!=0 else None, self.clu_weight))

            return pred, loss, hidden if mu is None else mu


        else:
            print("The task name {} is undefined in tasks_config!".format(task))
            exit(0)
