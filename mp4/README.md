# README for MP3

## How to run the project:

1. run FMaster.py at virtual machine 1.
     ```
    python3 FMaster.py
   ```
2. run coordinator.py at virtual machine 2.
    ```
   python3 coordinator.py
   ```
3. continue to join all the other node., by running file_server.py at them

## commands are listed as follow:
- put localfilename sdfsfilename
- get sdfsfilename localfilename
- delete sdfsfilename
- get_versions sdfsfilename num_versions localfilename
- ls sdfsfilename
- store
Upload validation files:
- utd
- set_batch_size model new_batch_size
- run_job model
Print stats on coordinator:
- get_stats