
![a robot](https://serve-md.charlesdlandau.net/img/robot-rrmonkey-unsplash.jpg)

#### Train Robots to Shoot Lazers at Aliens With Software Funded By Elon Musk and Microsoft

In this introduction I'm going to skip ahead to the robots shooting lazers because life is too short. Don't be intimidated by the code below, if you stick around after the lazers I'll explain everything that's happening.


```python
import gym
from gym import wrappers
```


```python
class RandomAgentContainer:
    """A model that takes random actions and doesn't learn"""
    
    def __init__(self):
        pass
    
    def decision_function(self, env):
        return env.action_space.sample()
    
    def train_on_history(self, history):
        pass
model = RandomAgentContainer()
```


```python
import gym
import io
import base64
from IPython.display import HTML

def run_job(env_name, model, steps, episodes_per_train=1,
            verbose=False, video_dir=False):
    env = gym.make(env_name)
    
    # Video output
    if not video_dir:
        video_dir = "./gym-results"
    
    env = wrappers.Monitor(env, video_dir, force=True)
    
    # initial observation & reset
    observation = env.reset()
    
    history = []
    
    episode = 0
    
    for i in range(steps):
        action = model.decision_function(env)
        observation, reward, done, info = env.step(action)
        history.append({
            "episode": episode,
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
            "action": action,
            "step": i
        })
        if done:
            if verbose:
                print(f"Done with episode {episode}, {info}")
            
            # Train model
            if episodes_per_train and episode % episodes_per_train==0:
                model.train_on_history(history)
            
            
            episode += 1
            env.reset()
    env.close()
    data = {
        'history': history,
        'env':env,
        'parameters': {
            'steps': steps,
            'env_name': env_name,
            'episodes_per_train': episodes_per_train,
            'video_dir': video_dir
        }
        
    }
    return data

result = run_job("SpaceInvaders-v0", model,
        1000, episodes_per_train=0);
```


```python
def render_video(episode_num, env, video_dir=None):
    if not video_dir:
        video_dir = './gym-results'
    video_file_handle = '{}/openaigym.video.{}.video{:0>6}.mp4'.format(video_dir, env.file_infix, episode_num)
    with io.open(video_file_handle, 'r+b') as file:
        video = file.read()
        encoded = base64.b64encode(video)

        return HTML(data='''
        <video width="360" height="auto" alt="test" controls><source src="data:video/mp4;base64,{0}" type="video/mp4" /></video>'''
    .format(encoded.decode('ascii')))
    
render_video(0, result['env'])
```

[Voilà!](https://serve-md.charlesdlandau.net/img/openai-spaceinvaders-vid.mp4)

And that's all it took! We wrote some code and it made the robot shoot the lazers. If that's what you came for, now's the time to leave. *But*, if you're interested in adding reinforcement learning to your toolbox, I hope you'll read on. And now we'll get to the topic behind the topic...

#### Introduction to Reinforcement Learning

I'm going to devote some time to writing about beginner/intermediate RL solutions starting with a framework called OpenAI Gym. OpenAI (and Gym) have been around for years as Elon Musk and Sam Altman's AI startup, but it has [just now](https://twitter.com/satyanadella/status/1153351797223768068?s=20) come out that OpenAI will colab with Microsoft's public cloud. OpenAI is still actively maintained and adding features at time of writing. In this introduction we're going to unpack basic concepts and explore the code you saw above line-by-line to help understand OpenAI Gym.

**With that out of the way....** We're going to set aside the code you saw above for a little while and introduce core concepts at...

![spaceballs_ludicrous_speed](https://static.tvtropes.org/pmwiki/pub/images/ludicrous_speed.jpg)

#### Machine Learning Besides

I've seen too many articles titled something like "Machine Learning vs AI What's the Difference and Don't You CARE?" Let's get through this in as few words as possible, and with some serious oversimplifiying.

**Machine Learning/AI**: The machine predicts something that it hasn't seen before using math or something. Wow! -- All the other categories are subsets of this.

**Supervised Learning**: We hide something (or multiple things) from the machine and ask it to predict what we hid. Because we have the hidden stuff, we know when the machine is wrong and use that information to improve the performance of the predictor by "training" it.

**Unsupervised Learning**: We have data, but we don't have the column that we would hide from the machine in the supvised learning category. Nonetheless we ask the machine to remark on the data. Often this involves sorting the data into "clusters." 

**Reinforcement Learning**: We tell the machine "go drive a car lol." We give it a high-five and a reward if it doesn't crash (or whatever.) If it crashes we punish it somehow. It doesn't have to be a car: it could be a video game, or a robot, or an elastic compute instance on the public cloud, or a bunch of other things. The car is simulated -- maybe.

#### Hit the Gym

Let's quickly unpack the code I shared earlier:


```python
import gym
from gym import wrappers
```

OpenAI gym is a framework for managing the interactions, outputs, and rendering for **environments** and **agents**. We'll start by creating an agent:


```python
class RandomAgentContainer:
    """A model that takes random actions and doesn't learn"""
    
    def __init__(self):
        pass
    
    def decision_function(self, env):
        return env.action_space.sample()
    
    def train(self, o):
        pass
        
    def train_on_history(self, history):
        pass

model = RandomAgentContainer()
```

Our "random agent" takes a random action by sampling the `action_space`. For example, in a Nintendo game, the "action space" is basically just the collection of all buttons on the controller. I decided to standardize my agents to always have `train`, `train_on_history` and `decision_function` methods, but this agent never really trains since it always uses the same rule to make a decision.

```python
def run_job(env_name, model, steps, episodes_per_train=1,
            verbose=False, video_dir=False)
```

This is a helper function that standardizes how we train agents. OpenAI is somewhat unopinionated about how you train agents  so stuff like this is left up to us.

`env_name`: gets passed to `gym.make()` which wants a string like `SpaceInvaders-v0`.

`model`: an instance of one of our agent classes.

`steps`: how many times the environment will run the `step()` method (we'll explore that soon).

`episodes_per_train`: how many steps between each call to `train_on_history` for the agent?

`verbose`: is checked before printing anything.

`video_dir`: is where the renderings are stored. Defaults to "gym_results"

After being called, our helper function starts with:

```python
    env = gym.make(env_name)
```

You might call this the entrance to the gym: this creates the environment instance that we'll use throughout the rest of the function. The next couple lines take care of video recording settings:
```python
    # Video output
    if not video_dir:
        video_dir = "./gym-results"
    
    env = wrappers.Monitor(env, video_dir, force=True)
```

Then we do some initialization:
```python
    # initial observation & reset
    observation = env.reset()
    
    history = []
    
    episode = 0
```

By convention, we `reset` the environment before taking any other action. I also want the history of our steps, but this can be cumbersome in some environments so we'll disable or parameterize that in the future. Finally, I want a count of our episodes, since each episode gets its own video and JSON file with a name derived from that number (among other things.)

```python
    for i in range(steps):
        action = model.decision_function(env)
        observation, reward, done, info = env.step(action)
```
This is the main flow of a `step` in gym. 

1. The agent model makes a decision based on some information from the environment. 
2. The environment advances a "step" with that action.
3. The step produces a new **observation** showing the new state of affairs in the environment that resulted from the action.
4. The step applies a **reward function** to calculate the reward (positive or negative) that the model will get.
5. The step returns additional information about the environment including whether the environment is **"done"** and any other data in **info**.

In Space Invaders the `observation` is a numerical representation of a screen. In many RL use cases we're responsible for writing our own reward function to evaluate the actions of our models, but in OpenAI gym it's provided for us. The environment is "done" when we win or lose. Every full interaction with the environment is referred to as an "episode" so when we're "done" we move to the next episode.

All this happens while a handful of frames, or even just one frame, display in the video.

```python
        history.append({
            "episode": episode,
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
            "action": action,
            "step": i
        })
```
Next, I record a bunch of data about the step we just took. I'm skeptical that you would ever record all this, especially `observation` which is basically an image. In any case, this gives you an idea for how much information is available to you at each step.

Lastly, we have some special actions to take if we're `done`:
```python
        if done:
            if verbose:
                print(f"Done with episode {episode}, {info}")
            
            # Train model
            if episodes_per_train and episode % episodes_per_train==0:
                model.train_on_history(history)
            
            
            episode += 1
            env.reset()

```

We optionally `print`, we optionally `train`, we increment the `episode` counter, and we `reset` the environment. This is very much like hitting the RESET button on an old console.

```python
    env.close()
    data = {
        'history': history,
        'env':env,
        'parameters': {
            'steps': steps,
            'env_name': env_name,
            'episodes_per_train': episodes_per_train,
            'video_dir': video_dir
        }
        
    }
    return data
```

Once the loop is finished we close the environment and return all the data we collected. We can use that data for our video_rendering function, which is mainly just a bunch of I/O ops, parsing to a Jupyter-friendly object, and this important string manipulation:

```python
video_file_handle = '{}/openaigym.video.{}.video{:0>6}.mp4'.format(video_dir, env.file_infix, episode_num)
```
And this is just to conform our filepath to OpenAI's naming convention for files, ensuring we find the right one.

#### In Review

In this introduction you saw robots shoot lazers at aliens. What more do you want? Okay well you also:

1. Learned about RL and about how it differs from other ML learning methodologies
2. Learned about OpenAI and their Environment/Agent model
3. Took a brief tour through the execution pattern involved for an environment's `step`.
4. Saw the entire environment lifecycle: first `.make`, then `done`, `.reset`, and `.step` cycles, and finally `.close`.

Next time maybe we'll land on the moon or something. Thanks for reading!
