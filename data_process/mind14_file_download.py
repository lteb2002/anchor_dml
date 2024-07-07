from datasets import load_dataset
import librosa
import torch
from transformers import AutoFeatureExtractor, WhisperModel, WhisperProcessor
from datasets import load_dataset
import data_process.vgg_funs as vgg_funs

model = WhisperModel.from_pretrained("openai/whisper-base")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")

minds_14 = load_dataset("PolyAI/minds14", "all")  # for French
# to download all data for multi-lingual fine-tuning uncomment following line
# minds_14 = load_dataset("PolyAI/all", "all")
# see structure
print(minds_14)
# load audio sample on the fly
audio_input = minds_14["train"][0]["audio"]  # first decoded audio sample
intent_class = minds_14["train"][0]["intent_class"]  # first transcription
intent = minds_14["train"].features["intent_class"].names[intent_class]

print(audio_input)
print(intent_class)
print(intent)

total = minds_14['train'].num_rows
print(total)
total = 10

output_file = 'F:\\datasets\\mind14_vec.csv'

head = ",".join(['x' + str(x + 1) for x in range(1024)]) + ",label\n"
with open(output_file, 'w') as file:
    file.write(head)
    count = 0
    for i in range(total):
        audio_input = minds_14["train"][i]["audio"]  # first decoded audio sample
        intent_class = minds_14["train"][i]["intent_class"]  # first transcription
        label = minds_14["train"].features["intent_class"].names[intent_class]
        # inputs = feature_extractor(audio_input["array"], return_tensors="pt", sampling_rate=16000)
        # input_features = inputs.input_features
        waveform = audio_input["array"]
        # print(waveform)
        correct_sampling_rate = 16000
        waveform = librosa.resample(waveform, orig_sr=audio_input['sampling_rate'], target_sr=correct_sampling_rate)
        input_features = processor(waveform, sampling_rate=correct_sampling_rate, return_tensors="pt").input_features
        decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
        last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
        # shape = list(last_hidden_state.shape)
        # print(shape)
        y = torch.flatten(last_hidden_state, 1)
        s = vgg_funs.build_tensor_as_ln(y, label)
        file.write(s)
        file.flush()
        count += 1
        print(count)
