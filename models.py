from utils import DinoSeleniumEnv, show_image, IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT
import torch
from torch import nn
from torch.nn import functional as F
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.optimizers import Adam
# from keras.models import Sequential

class Agent(object):
    def __init__(self, env): # takes dino game environment as input
        self._env = env
        self._jump() # start the game with a jump
        self.actions = [self._nothing, self._jump, self._duck]

    def _jump(self):
        self._env.press_up()
    
    def _duck(self):
        self._env.press_down()
    
    def _nothing(self):
        pass # do nothing
    
    def perform_action(self, action_indx):
        self.actions[action_indx]()

    
    def is_crashed(self):
        return self._env.is_crashed()

DEFAULT_REWARD=0.1
GAMEOVER_PUNISHMENT=-1
class GameState(object):
    def __init__(self, agent, debug=True):
        self._agent = agent
        self._display = None
        if debug:
            self._display = show_image() # a coroutine method
            self._display.__next__()
    def get_state(self, action_indx):
        score = self._agent._env.get_score()
        reward = DEFAULT_REWARD
        is_gameover = False
        self._agent.perform_action(action_indx)
        screen = self._agent._env.grab_screen()
        if self._display is not None:
            self._display.send(screen)
        if self._agent._env.is_crashed():
            is_gameover = True
            reward = GAMEOVER_PUNISHMENT
            self._agent._env.restart_game()
        return screen, reward, is_gameover, score

class FlattenTorch(nn.Module):
    def forward(self,x):
        return x.view(x.shape[0], -1)

class QNetwork(nn.Module):
    def __init__(self, nb_actions, hidden_size=512):
        super(QNetwork, self).__init__()
        self.bn0 = nn.BatchNorm2d(IMAGE_CHANNELS)
        self.image_encoder = nn.Sequential(nn.Conv2d(IMAGE_CHANNELS, 32, 7, stride=4, padding=2),
                                            nn.MaxPool2d(2),
                                            nn.ReLU(),
                                            nn.Conv2d(32,64,4,stride=2,padding=1),
                                            nn.MaxPool2d(2),
                                            nn.ReLU(),
                                            nn.Conv2d(64,64,3,stride=1,padding=1),
                                            nn.MaxPool2d(2),
                                            nn.ReLU(),
                                            FlattenTorch())
        self.q_value_estimator = nn.Sequential(nn.Linear(64,hidden_size),
                                                nn.ReLU(),
                                                nn.Linear(hidden_size,nb_actions))
    
    def forward(self, image):
        if next(self.parameters()).is_cuda:
            image = image.cuda()
        image = self.bn0(image)
        encoded_features = self.image_encoder(image)
        return self.q_value_estimator(encoded_features)


# def build_keras_model():
#     print("Now we build the model")
#     model = Sequential()
#     model.add(Conv2D(32, (8, 8), padding='same',strides=(4, 4),input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS)))  #80*80*4
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Activation('relu'))
#     model.add(Conv2D(64, (4, 4),strides=(2, 2),  padding='same'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Activation('relu'))
#     model.add(Conv2D(64, (3, 3),strides=(1, 1),  padding='same'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Activation('relu'))
#     model.add(Flatten())
#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dense(2))
#     adam = Adam(lr=1e-4)
#     model.compile(loss='mse',optimizer=adam)
    
#     return model