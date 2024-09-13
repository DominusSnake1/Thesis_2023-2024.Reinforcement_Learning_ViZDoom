## System Information
OS: Ubuntu 22.04.3 LTS<br>
GPU: AMD® Radeon RX 6900 XT 16 GB<br>
CPU: Intel® Core™ i7-13700K<br>
RAM: 32 GB DDR5 <br>

## <u>Step 0</u>:<br>Create a Conda Environment (if it doesn't exist).
```bash
conda create --name DoomPy39 python=3.9 pip
```

## <u>Step 1</u>:<br>Activate the Conda Environment.
```bash
conda activate DoomPy39
```

## <u>Step 2</u>:<br>Install the required packages.
```bash
pip3 install -r ./Other/requirements.txt
```

## <u>Step 3</u>:<br>This is required for the Project to work.
```bash
git clone https://github.com/Farama-Foundation/ViZDoom.git
```

## <u>Step 4</u>:<br>Run the PyTorch/ROCM test script.
```bash
python ./Other/test_rocm.py
```

## <u>Step 5</u>:<br>Run the program in either training or testing mode.
You can also view real-time graphs of the training process using TensorBoard.
```bash
gnome-terminal -- sh -c 'cd Data/Logs/ | tensorboard --logdir=.'
```
___
The base command for the terminal is:
`main.py -lvl LEVEL -m MODE -t TECHNIQUE [-mdl MODEL] [-eps EPISODES] [-r] [-d]`

If help is needed, type `main.py -h` or `main.py --help` for a quick explanation of the commands.

### Example Usage:
#### * Training Mode.
```bash
python ./main.py -lvl SELECT_LEVEL -m train -t YOUR_TECHNIQUE -r -d
```

There is also a script included to run and train all scenarios with all Techniques.
```bash
./superset.sh
```
#### * Testing Mode.
(Pick a model from `Data/Train/PICK_YOUR_LEVEL`)
```bash
python ./main.py -lvl SELECT_LEVEL -m test -t YOUR_TECHNIQUE -mdl YOUR_MODEL -eps X -r -d
```

## Display the map layout.
In case the layout of the scenario map is needed:
```bash
python ./Other/DisplayMapLayout.py CHOOSETHESCENARIO 
```

## Get the Agent's First-Person view.
In case the agent's POV is needed of the scenario:
```bash
python ./Other/DisplayFirstPersonView.py CHOOSETHESCENARIO
```
