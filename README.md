# Neolithic Agent-Based Simulation

This is a Python-based agent-based modeling (ABM) tool for simulating villager behavior, land use, and inequality in a small-scale village environment in the Neolithic societies context. 

The simulation supports configurable demographic and economic parameters, and produces visual and numerical outputs.

![Web Demo](app/demo_web.png)

---

## Overview

Villagers (agents) and households in the simulation manages land and food, may engage in farming, trade, marriage, and migration, and evolves over time. The model allows users to simulate:
- Population growth and decline
- Land regeneration and degradation
- Optional trading, fallow farming, and emigration mechanisms
- Inequality development over time (tracked via the Gini coefficient)

The simulation runs through a web interface built with Flask, and outputs include:
- A simulation animation (`.gif`)
- Population and Gini plots (`.svg`)
- Raw data in `.csv` format

---

## Requirements

Clone this repository and nevigate to the folder:

```bash
git clone https://github.com/yutongliang723/neolithic_abm_simulator.git
cd neolithic_abm
```

This tool requires **Python 3.12.7**.

You can open a terminal and check your current version with:

```bash
python --version
```

Create and activate a python3.12 environment:
```bash
python3.12 -m venv neoabm
source neoabm/bin/activate
```
Now, you should see the environment activated in front of the repository. (neoabm) xxx

Dependencies include:

- Flask
- pandas
- matplotlib
- numpy
- scipy
- Pillow  
....

To install everything:

```bash
pip install -r requirements.txt
```

Start the web application

```bash
python app/interaction.py
```

Open your browser and go to

```
http://localhost:5001
```

Run test

```
python -m unittest tests/test_utils.py
```
or 
```
python -m unittest discover tests
```

If all tests passed, then you will see similar outputs

```.
---------------------------------------
Ran 1 test in 0.002s

OK

```

After entering the desired parameters and executed the experiment by clicking "Run Simulation", the result similar to the image below can be expected.

![Web Demo](app/demo_web2.png)