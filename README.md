# ASR project barebones

## Installation guide

On your fist step run installations of requirements (I used python3.8)
```shell
pip install -r ./requirements.txt
```
Second step is to load dataset

```shell
bash loader.sh
```
After it you can run train on your config by command:
```shell
python3 train.py -c PATH/TO/YOUR/CONFIG/CONFIG_NAME.json 
```
I ran with tran.json config:
```shell
python3 train.py -c tts/configs/train.json
```

To test my model you may run:
```shell
bash loader.sh
```
After downloading waveglow and checkpoint you may run following:
```shell

python3 test.py -i default_test_model/text.txt \
   -r model_info/checkpoint_epoch.pth \
   -w waveglow/pretrained_model/waveglow_256channels.pt \
   -o output_folder
```
Where -i has text on what we want to test.  -r has direction to checkpoint and you need to add in the same folder config with name config.json. -w is path to waveglow -o folder to write a file. You will get on every text wavs with format "{number_of_text}-{alpha}-{beta}-{gamma}.wav".
To download model you need take it from https://disk.yandex.ru/d/l2z1dQM1ziFWHQ.
