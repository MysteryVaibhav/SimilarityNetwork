import torch.utils.data
import torch.nn as nn
from model import SimilarityNetwork
from timeit import default_timer as timer
from util import *
import sys
from tqdm import tqdm


def init_xavier(m):
    """
    Sets all the linear layer weights as per xavier initialization
    :param m:
    :return: Nothing
    """
    if type(m) == torch.nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)
        m.bias.data.zero_()


class MarginLoss(nn.Module):
    """
    Class for the margin loss
    """

    def __init__(self, margin):
        super(MarginLoss, self).__init__()
        self.margin = margin

    def forward(self, s_u_v_w, s_v_w_, s_u_w_):
        loss = ((self.margin - s_u_v_w + s_v_w_).clamp(min=0) +
                (self.margin - s_u_v_w + s_u_w_).clamp(min=0)).sum()
        return loss


class Trainer:
    def __init__(self, params, data_loader, evaluator):
        self.params = params
        self.data_loader = data_loader
        self.evaluator = evaluator

    def train(self):
        model = SimilarityNetwork(self.params)
        model.apply(init_xavier)
        #model.load_state_dict(torch.load('models/model_weights_5.t7'))

        # Logistic loss
        loss_function = nn.SoftMarginLoss()
        loss_function.size_average = False

        if torch.cuda.is_available():
            model = model.cuda()
            loss_function = loss_function.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.params.lr, weight_decay=self.params.wdecay)
        try:
            prev_best = 0
            for epoch in range(self.params.num_epochs):
                iters = 1
                losses = []
                start_time = timer()
                num_of_mini_batches = len(self.data_loader.train_ids) // self.params.batch_size
                for caption, mask, image, label in tqdm(self.data_loader.training_data_loader):

                    model.train()
                    optimizer.zero_grad()
                    # forward pass.

                    # Train using phrases minibatch
                    if torch.cuda.is_available():
                        caption = caption.cuda()
                        mask = mask.cuda()
                        image = image.cuda()
                        label = label.cuda()
                    sim = model(torch.autograd.Variable(caption), torch.autograd.Variable(mask),
                                       torch.autograd.Variable(image), False, True)

                    loss = loss_function(sim, torch.autograd.Variable(label))

                    # Sum both the losses
                    loss.backward()
                    losses.append(loss.data.cpu().numpy())
                    if self.params.clip_value > 0:
                        torch.nn.utils.clip_grad_norm(model.parameters(), self.params.clip_value)
                    optimizer.step()

                    #                     sys.stdout.write("[%d/%d] :: Training Loss: %f   \r" % (
                    #                         iters, num_of_mini_batches, np.asscalar(np.mean(losses))))
                    #                     sys.stdout.flush()
                    iters += 1

                if epoch + 1 % self.params.step_size == 0:
                    optim_state = optimizer.state_dict()
                    optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / self.params.gamma
                    optimizer.load_state_dict(optim_state)

                torch.save(model.state_dict(), self.params.model_dir + '/model_weights_{}.t7'.format(epoch + 1))

                # Calculate r@k after each epoch
                if (epoch + 1) % self.params.validate_every == 0:
                    r_at_1, r_at_5, r_at_10 = self.evaluator.recall_phrase_localization(model, is_test=False)

                    print("Epoch {} : Training Loss: {:.5f}, R@1 : {}, R@5 : {}, R@10 : {}, Time elapsed {:.2f} mins"
                          .format(epoch + 1, np.asscalar(np.mean(losses)), r_at_1, r_at_5, r_at_10,
                                  (timer() - start_time) / 60))
                    if r_at_1 > prev_best:
                        print("Recall at 1 increased....saving weights !!")
                        prev_best = r_at_1
                        torch.save(model.state_dict(),
                                   self.params.model_dir + 'best_model_weights_{}_{:.3f}.t7'.format(epoch + 1, r_at_1))
                else:
                    print("Epoch {} : Training Loss: {:.5f}".format(epoch + 1, np.asscalar(np.mean(losses))))
        except KeyboardInterrupt:
            print("Interrupted.. saving model !!!")
            torch.save(model.state_dict(), self.params.model_dir + '/model_weights_interrupt.t7')


