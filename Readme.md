# Play Dino Run with RL implemented in Pytorch
After reading [this](https://blog.paperspace.com/dino-run/) amazing blog about training Deep RL of QNetwrok for learning [Dino](https://chromedino.com/). I decided to implemented it myself in [Pytorch](https://pytorch.org).
## How to play
You need to install the depenedenceis and run the `train.py` with the apporopriate arguements.
### Dependencies
First you need to download chrome driver from [here](http://chromedriver.chromium.org/) and extract it in a directory and use its path for the `chrome_driver_path` arguemnt.  
Install requirements of the project:  
* pytorch: go to pytorch.org for install instruction
* numpy: pip install numpy
* selenium: pip install selenium
* Pillow: pip install Pillow
* opencv-python: conda install -c conda-forge opencv
Or install them with `requirements.txt`.

### Training and testing
You can train or play with the dino with `train.py` scripts and its arguemnents which are enumerated as follow:   
```bash
--chrome_driver_path CHROME_DRIVER_PATH
                        Path of chrome driver
  --checkpoint_path CHECKPOINT_PATH
                        Path of Pytorch model path
  --nb_actions NB_ACTIONS
                        Number of possible actions for Dino. Default is 2 but
                        you can increase to 3 with do_nothing,jump and dive
                        actions.
  --initial_epsilon INITIAL_EPSILON
                        Starting epsilon value for explorations
  --final_epsilon FINAL_EPSILON
                        Final value for epsilon after exploration
  --gamma GAMMA         Value of gamma for atenuation of rewards in next
                        states.
  --nb_memory NB_MEMORY
                        Number of memory to store previous states and rewards
                        for training.
  --nb_expolre NB_EXPOLRE
                        Number of times for explorations. After this time the
                        epsilon is in final_epsilon value and the explorations
                        is in its minumum value.
  --is_debug            A flag for debugging. If enabled an OpenCV window is
                        shown that illustrates the feeded image to the netwrok
  --batch_size BATCH_SIZE
                        Number of batch for training.
  --nb_observation NB_OBSERVATION
                        Number of observation before starting training
  --use_cuda            Use cuda if it\'s avaulable
  --exploiting          If enabled, there is no training the qnetwork just
                        predict the qvalues.
  --log_frequency LOG_FREQUENCY
                        Frequency of logging every time step.
  --save_frequency SAVE_FREQUENCY
                        Frequency of saving satate of the training
  --game_speed GAME_SPEED
                        Speed of the game.
  --ratio_of_win RATIO_OF_WIN
                        Ration of usage of win actions in training. It should
                        be between (0,1]. 1 means use all actions and 1e-6
                        means small amount of win actions.
  --desired_fps DESIRED_FPS
                        If you want to reduce processing fps in order to have
                        constant fps in training and testing time.
```

## Things to try
There are a number of things in which one can try and test to see for herself!  
You can change model of QNetwork or play with the arguments. Here I illustrate some of my trials:
* Train from scratch with default values (This is almost the same as the [DeepMind paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) in 2013)
    * python train.py --chrome_driver_path /path/to/chromedriver
* Train more with failures. One can assume a human can learn more from its failures than its sucesses. You can decrease the amount of train data which is for winning actions by `ratio_of_win=0.1`
    * python train.py --chrome_driver_path /path/to/chromedriver --ratio_of_win 0.1
* You can change the speed of the game and train it faster but it requires a better CPU/GPU machine
    * python train.py --chrome_driver_path /path/to/chromedriver --game_speed 3
* You can see a sample almost trained network playing Dino without learning or exploration
    * python train.py --chrome_driver_path /path/to/chromedriver --checkpoint_path freezed_model.pth --exploiting

## Sample result after 400 epsiodes of game:
Here is a sample result of the agent after training with 400 episodes of game:  
The trained model exists in this repository under the name of `freezed_model.pth`  
![Trained Dino](out.gif)
