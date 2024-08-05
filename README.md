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
### * Training Mode.
You can also view real-time graphs of the training process by navigating to `Data/Logs/PICKYOURLEVEL/SELECTPPO`.
```bash
gnome-terminal -- sh -c 'cd Data/Logs/ | tensorboard --logdir=.'
```
```bash
python ./main.py -lvl SELECT_LEVEL -m train -t YOUR_TECHNIQUE -d -r
```
### * Testing Mode. (Pick a model from `Data/Train/PICKYOURLEVEL`)
```bash
python ./main.py -lvl SELECT_LEVEL -m test -t YOUR_TECHNIQUE -mdl YOUR_MODEL -eps X
```
