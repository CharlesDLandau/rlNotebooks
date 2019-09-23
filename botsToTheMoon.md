
### 🤖➡🌜 (Teach Bots to Land on the Moon)

[Tinker with this notebook in Google Colab](https://colab.research.google.com/github/CharlesDLandau/rlNotebooks/blob/master/botsToTheMoon.ipynb)

[GitHub](https://github.com/CharlesDLandau/rlNotebooks)

In the past few episodes we've gone over design princples for reinforcement learning, introduced reinforcement learning with OpenAI Gym, and trained a deep learning model to navigate an OpenAI Gym environment. To accomplish this, we leveraged the agent-environment relationship:

* The agent takes actions in the environment. These can be random, or guided by heuristics, or by a model, or some combination. It recieves data about the environment, like a picture of the environment or a string, and then it takes an action. Afterwards, it recieves a reward from the environment, which amount to positive or negative "reinforcement" of the selected behavior.

* The environment tracks the state, possible actions, rewards and **reward function**, and optionally renders the environment to a display. It receives actions from the agent, and it can be configured to persist data (e.g. a video of the agent performing in the environment.)

#### Better Agents

In previous episodes we also implemented some janky, simplistic reinforcement learning. Without convolutional layers or a policy engine the network could only learn simplistic strategies (like [turn left](https://dev.to/charlesdlandau/drive-a-tank-with-python-3caf)).

To solve this without reinventing the wheel we can use libraries like [Tensorflow Agents](https://github.com/tensorflow/agents), or [OpenAI Baselines](https://github.com/openai/baselines) which have some battle-tested implementations of the most successful agents. I've worked with a few of these libraries and I mostly came to the same conclusions as this [excellent article](https://medium.com/data-from-the-trenches/choosing-a-deep-reinforcement-learning-library-890fb0307092) by Thomas Simonini. Namely, I like [Stable Baselines](https://github.com/hill-a/stable-baselines), a fork of the OpenAI Baselines library.

#### Getting Ready

![Getting Ready](https://thumbs.gfycat.com/GorgeousUnfortunateAndalusianhorse-size_restricted.gif)

Setting up Google Colab with Stable Baselines and its dependencies, and figuring out how to get gifs working, took me longer than I would have liked. You do it with the following lines:


```python
# see: colab.research.google.com/drive/1GLlB53gvZaUyqMYv8GmZQJmshRUzV_tg
!apt-get -qq -y install libcusparse8.0 libnvrtc8.0 libnvtoolsext1 > /dev/null
!ln -snf /usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so.8.0 /usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so
!apt-get -qq -y install xvfb freeglut3-dev ffmpeg> /dev/null

!pip install stable-baselines==2.7.0

# box2d fork until SWIG dependency is replaced
!pip install box2d-py

# pyglet later versions break gym apparently
!pip install pyglet==1.3.2

# Set up display; otherwise rendering will fail for
# classic control type envs, apparently
import os
import Box2D
import pyglet
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'


# See https://github.com/ipython/ipython/issues/10045#issuecomment-522697219
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from IPython import display
```

With that out of the way, you can implement a high-quality Advantage Actor Critic (A2C) model in just two lines. Simple baselines is great because it has intuitive methods for training and predicting. Even better, it stores our enviornment as a `model.env` member, so all we need for typical workflows is a `model` instance.

Let's train -- The following takes a fair bit of time to run:


```python
from stable_baselines import A2C
import time
start = time.time()

# Train a multilayer perceptron in the lunar lander.
model = A2C('MlpPolicy', "LunarLander-v2").learn(900000)

# Mine finished in about 10 minutes.
print(f"Finished in {time.time()-start}s")
```

### An actor and a critic walk into a bar.

While that runs, let's talk about the Actor-Critic paradigm. The problem Actor-Critic solves involves none other than our reward function. In many cases the rewards only happen at the end of an "episode" AKA a "game." Consider a reward function for tic-tac-toe:

* Return 0 during play
* Return 0 if game ends in a tie
* Return -1.0 if game ends in a loss for the agent
* Return 1.0 if game ends in a win for the agent

As you can see, our agent is taking several steps each episode which recieve no feedback at all from the reward function, slowing training and making it difficult to give fine-grained feedback to our agent about specific actions. To solve this, Actor-Critic actually implements two models:

*  A Critic model that tries to predict reward values that a given action will earn, given an observation about the environment. The table of predicted reward values for action-observation pairs is sometimes called a "Q table."
*  An Actor model that takes observations from the environment, but not rewards. It learns a strategy for playing in the environment, AKA a "policy." Instead of optimizing on the environment's rewards directly, this model optimizes on _feedback from the critic model_. When we do predictions, just the Actor model needs to be called.

We can easily imagine our Actor model in the lunar lander module, while the Critic sits in the command center observing the lander's actions.

![the critic](https://thumbs.gfycat.com/FlawlessWhichAdamsstaghornedbeetle-size_restricted.gif)

Each time an action is taken the Critic consults a table of actions, angles, speeds, and so on. Seeing whether the action will contribute to a successful landing, the Critic radios over to the Actor with feedback about the action, and the Actor learns to be a better pilot from the feedback. Over the course of thousands of landings, the Critic updates the table frequently based on crashes, successes, and surprises. Some days, the Actor flies with the radio turned off, but it doesn't learn anything.

### Record a GIF of the landing

By now your model might be done training. Per the docs, here's how we record a gif of our model in action:



```python
import imageio
import numpy as np

# Number of images to capture
n_images = 1200

# Our array of images
images = []

# init a new environment
obs = model.env.reset()
# init the img var with the starting state of the env
img = model.env.render(mode='rgb_array')

for i in range(n_images):
  # At each step, capture an image
  images.append(img)

  # Advance a step an render
  action, _ = model.predict(obs)
  obs, _, _ ,_ = model.env.step(action)
  img = model.env.render(mode='rgb_array')

imageio.mimwrite('./lander.gif',
                [np.array(img) for i, img in enumerate(images) if i%2 == 0],
                fps=29)
```

If you're successful then `lander.gif` should look something like this:

![lander.gif](https://serve-md.charlesdlandau.net/img/lander.gif)

Displaying gifs in Jupyter is a little counterintuitive, here's how I do it:


```python
from pathlib import Path
gifPath = Path("lander.gif")
# Display GIF in Jupyter, CoLab, IPython
with open(gifPath,'rb') as f:
    display.Image(data=f.read(), format='png')
```

### Wrapping Up...

In this episode you:

* Learned about Actor-Critic
* Learned about Simple Baselines
* Saw how to setup giffing the environments in a notebook
* Landed on the moon!

![goodjob.gif](https://thumbs.gfycat.com/PerfumedBelatedBlowfish-size_restricted.gif)

Way to go!

In future episodes we'll take a closer look at how RL can create value for users, businesses, and robot overlords. Stay tuned and thanks for reading.




