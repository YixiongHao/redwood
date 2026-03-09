# Reflection

Thanks for making this! One of the most fun work tests I've taken.

Note: the countdown for test phase on your website is 2 hours, not 15 mins!

## Play-by-Play

A play by play of how I went about the problem (to the best of my memory):

- Vibed all of it, didn't write a single line of code, this felt great. uplift is real!
- Implemented baseline best of n with basic prompting just to understand what the problem asks and how Qwen reacts.
- Spent some time researching code golfing (never heard of it before) and c++ strategies.
- I know of a few ways to make LLMs better at problems, best of N, iterate on best answers, in-context RL, few shot, etc. But no clue which ones would work best here. So I launch a bunch of Claude research agents to find ways to best leverage async compute.
- Moved to a 3 stage pipeline (this is a good description of the final scaffold):
  - Best of N to find the correct solution with basic golfing. Samples with a variety of prompts and temperatures.
  - Multi round best of N shortening with few shot examples and prompting based on researching code golfing tips.
  - Multi round best of N evolution that asks the LLM to combine the best tips and tricks from two candidate solutions.
- Then, tried to look at stage over stage improvements to tune hyperparameters, and piled on a bunch of prompt based optimizations based on vibes. "yolo" runs if you will.
  - Introduced a failsafe catch that uses a bit of 'in context RL' when Phase 1 fails at finding a correct solution by appending wrong attempts.
- Finish!

## Pivots

- Abandoned several prompting practices that made the model worse because it's dumb and can't count characters.
- Had one unfortunate bug where python standard string parsing interpreted some C++ golfing hack syntax the wrong way. This took away 1-2 feedback loops away from me :(

## Budget Strategy

I only managed to spend 7/50 dollars... definitely fumbled here. I briefly thought about aggressively increasing budgets but didn't want evaluation time to exceed the 15 mins (it was already taking 10 min to run). This was definitely the wrong move because I forgot about the hint of bottlenecked on server responses, not inference. **I ALSO FORGOT TO TURN ON THINKING FOR PHASES 2 AND 3 NOOOOO**. Too much vibing, not enough controlling. Please cut me some slack here haha I did this at 2am.

## If I Had 2 Days and More Money

I would've tried a RLLM (recursive LLM) esque approach that builds an actual agent that can interact with each individual problem, send multiple copies to do things in parallel, to gather information, etc. This probably better leverages TTS and makes it such that the agent can enter its own feedback loops. I didn't go with this because I wasn't super familiar with this version of Qwen and didn't think I can execute and iterate on this in 2 hours.

## Information Value

I had Claude build itself a bunch of scripts and skills it can use to look at saved rollouts and compute metrics of interest to report to me. I built this half way through one of the runs, but building this from the start probably would've given me 1-2 more complete feedback loops.

## Gambles

Going with the 'find local optimum of a conservative strategy' given the time limit. In expectation, this paid off. I guess we'll have to see.

## Assumptions and Mistakes

- Forgetting that I'm server response bottlenecked... in hindsight the hyperparam tuning probably didn't matter lol.
- Not double checking subtle design choices that matter, but Claude misses, like turning on reasoning. I do this a lot when using LLM agents to help code.
- I should've added a stage 4 of the pipeline that integrates information from the previous 3 in the context and asked for final optimizations. An obvious choice I ran out of time to implement due to other time-consuming mistakes.
- I assumed that a hand crafted TTS pipeline would work better than pure in-context RL with a basic agent scaffold that lets Qwen call tools to test its solutions. My impression is that the Qwen model we use isn't smart enough for this, but I still don't know whether this is true unless tested.
