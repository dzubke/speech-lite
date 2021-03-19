# standard libraries
from typing import Dict, List
# third party libraries
import torch





class TrainingState:
    def __init__(self,
                 model_state:dict=None,
                 optim_state:dict=None,
                 avg_loss:float=0,
                 iteration:int=0,
                 last_epoch:int=0,
                 best_per:float=None,
                 dev_loss: Dict[str, List[float]], 
                 dev_per: Dict[str, List[float]],
                 config:dict=None):
        """

        """
        self.model_state = model_state
        self.optim_state = optim_state
        self.avg_loss = avg_loss
        self.iteration = iteration
        self.last_epoch = last_epoch
        self.best_per = best_per
        self.dev_loss = dev_loss
        self.dev_per = dev.per


    def update(self, name_val_dict:dict):
        # checks that the update names (keys in name_val_dict) are all
        # attribute names of the train_state object
        attr_names = self.__dict__.keys()
        wrong_names = []
        for update_name in name_val_dict.keys():
            if update_name not in attr_names:
                wrong_names.append(update_name)
        assert len(wrong_names)== 0,\
            f"update names {wrong_names} don't match train_state attribute names"

        # update the object's attributes
        self.__dict__.update(name_val_dict)



    def serialize_state(self, epoch, (iteration, avg_loss) ):
        return {
            'model_state_dict': self.model_state
            'optim_dict': self.optim_state,
            'avg_loss': self.avg_loss,
            'iteration': iteration,
            'last_epoch': epoch,
            'best_per': self.best_per,
            'dev_loss': self.dev_loss,
            'dev_per': self.dev_per
        }

    @classmethod
    def load_state(cls, state_path):
        print("Loading state from model %s" % state_path)
        state = torch.load(state_path, map_location=lambda storage, loc: storage)
        
        avg_loss = int(state.get('avg_loss', 0))
        loss_results = state['loss_results']



        return cls(model_state = state['model_state_dict']
                   optim_state=state['optim_dict'],
                   best_per=state['best_per'],
                   avg_loss=avg_loss,
                   epoch=epoch,
                   training_step=training_step)

    def save_checkpoint(self, epoch, state, i=None):
        if self.save_n_recent_models > 0:
            self.check_and_delete_oldest_checkpoint()
        model_path = self._create_checkpoint_path(epoch=epoch,
                                                  i=i)
        print("Saving checkpoint model to %s" % model_path)
        torch.save(obj=state.serialize_state(epoch=epoch,
                                             iteration=i),
                   f=model_path)



    

    def print_attr_names(self):
        return " ".join(list(self.__dict__.keys()))
    

    def add_results(self, epoch:int, dev_loss:dict, dev_per:dict):

        for dev_name, dev_loss_value in dev_loss.items():
            self.dev_loss.get(dev_name).append(dev_loss_value)
        
        for dev_name, dev_per_value in dev_per.items():
            self.dev_per.get(dev_name).append(dev_per_value)
    
    def reset_avg_loss(self):
        self.avg_loss = 0

    def _reset_amp_state(self):
        self.amp_state = None

    def _reset_optim_state(self):
        self.optim_state = None

    def _reset_epoch(self):
        self.epoch = 0


#class ResultState:
#    def __init__(self, 
#                 loss_results:Dict[str, List[float]], 
#                 per_results:Dict[str, List[float]]):
#        self.dev_loss = loss_results
#        self.dev_per = per_results
#
#    def add_results(self,
#                    epoch,
#                    loss_result,
#                    wer_result,
#                    cer_result):
#        self.loss_results[epoch] = loss_result
#        self.wer_results[epoch] = wer_result
#        self.cer_results[epoch] = cer_result
#
#    def serialize_state(self):
#        return {
#            'loss_results': self.loss_results,
#            'wer_results': self.wer_results,
#            'cer_results': self.cer_results
#        }

