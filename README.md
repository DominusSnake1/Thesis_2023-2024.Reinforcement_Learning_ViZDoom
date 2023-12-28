## System Information
OS: Ubuntu 22.04.3 LTS<br>
GPU: AMD® Radeon RX 6900 XT<br>
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
pip install -r requirements.txt
```
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

## <u>Step 3</u>:<br>This is required for the Project to work.
```bash
git clone https://github.com/Farama-Foundation/ViZDoom.git
```

## <u>Step 4</u>:<br>Run the PyTorch/ROCM test script.
```bash
python ./test_rocm.py
```

## <u>Step 5</u>:<br>Run the program.
```bash
python ./main.py
```