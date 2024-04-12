import argparse
import os
import subprocess

import numpy as np
import torch
import yaml
from scipy.io import wavfile

from mtts.models.fs2_model import FastSpeech2
from mtts.models.vocoder import *
from mtts.text import TextProcessor
from mtts.text.gp2py import TextNormal
from mtts.utils.logging import get_logger

logger = get_logger(__file__)


def check_ffmpeg():
    r, path = subprocess.getstatusoutput("which ffmpeg")
    return r == 0


with_ffmpeg = check_ffmpeg()


def build_vocoder(device, config):
    vocoder_name = config['vocoder']['type']
    VocoderClass = eval(vocoder_name)
    model = VocoderClass(**config['vocoder'][vocoder_name])
    return model


def normalize(wav):
    assert wav.dtype == np.float32
    eps = 1e-6
    sil = wav[1500:2000]
    #wav = wav - np.mean(sil)
    #wav = (wav - np.min(wav))/(np.max(wav)-np.min(wav)+eps)
    wav = wav / np.max(np.abs(wav))
    #wav = wav*2-1
    wav = wav * 32767
    return wav.astype('int16')


def to_int16(wav):
    wav = wav = wav * 32767
    wav = np.clamp(wav, -32767, 32768)
    return wav.astype('int16')


def tts(input_text,checkpoint,style="0",device='cuda',output_dir="./outputs/",duration=1.0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_dir,file_name = os.path.split(os.path.realpath(__file__))
    config_path = os.path.join(file_dir,"config.yaml")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
        logger.info(f.read())
   
    config['hanzi_embedding']['vocab']=os.path.join(file_dir,'gp.vocab')
    config['pinyin_embedding']['vocab']=os.path.join(file_dir,'py.vocab')
    config['vocoder']['VocGan']['checkpoint']=os.path.join(file_dir,"vctk_pretrained_model_3180.pt")
    config['dataset']['train']['emb_type1']['vocab']=os.path.join(file_dir,'py.vocab')
    config['dataset']['train']['emb_type2']['vocab']=os.path.join(file_dir,'gp.vocab')
    sr = config['fbank']['sample_rate']
   
    vocoder = build_vocoder(device, config)
    text_processor = TextProcessor(config)
    model = FastSpeech2(config)

    if checkpoint != '':
        sd = torch.load(checkpoint, map_location=device)
        if 'model' in sd.keys():
            sd = sd['model']
    model.load_state_dict(sd)
    del sd  # to save mem
    model = model.to(device)
    torch.set_grad_enabled(False)
    
    
    input_text
    
    tn = TextNormal(config['hanzi_embedding']['vocab'], config['pinyin_embedding']['vocab'], add_sp1=True, fix_er=True)
    py_list, gp_list = tn.gp2py(input_text)
    text_sets = []
    for i,(py,gp) in enumerate(zip(py_list, gp_list)):
        # print(f'text{i}|'+py + '|' + gp+'|0')
        text_sets.append(f'text{i}|'+py + '|' + gp+f'|{style}')
    # try:
    #     lines = open(args.input).read().split('\n')
    # except:
    #     print('Failed to open text file', args.input)
    #     print('Treating input as text')
    #     lines = [args.input]
    lines = text_sets
    for line in lines:
        if len(line) == 0 or line.startswith('#'):
            continue
        logger.info(f'processing {line}')
        name, tokens = text_processor(line)
        tokens = tokens.to(device)
        seq_len = torch.tensor([tokens.shape[1]])
        tokens = tokens.unsqueeze(1)
        seq_len = seq_len.to(device)
        max_src_len = torch.max(seq_len)
        output = model(tokens, seq_len, max_src_len=max_src_len, d_control=duration)
        mel_pred, mel_postnet, d_pred, src_mask, mel_mask, mel_len = output

        # convert to waveform using vocoder
        mel_postnet = mel_postnet[0].transpose(0, 1).detach()
        mel_postnet += config['fbank']['mel_mean']
        wav = vocoder(mel_postnet)
        if config['synthesis']['normalize']:
            wav = normalize(wav)
        else:
            wav = to_int16(wav)
        dst_file = os.path.join(output_dir, f'{name}.wav')
        #np.save(dst_file+'.npy',mel_postnet.cpu().numpy())
        logger.info(f'writing file to {dst_file}')
        wavfile.write(dst_file, sr, wav)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='input.txt')
    parser.add_argument('--duration', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default='./outputs/')
    parser.add_argument('--checkpoint', type=str, required=True, default='')
    parser.add_argument('-c', '--config', type=str, default='./config.yaml')
    parser.add_argument('-d', '--device', choices=['cuda', 'cpu'], type=str, default='cuda')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.config) as f:
        config = yaml.safe_load(f)
        logger.info(f.read())

    sr = config['fbank']['sample_rate']

    vocoder = build_vocoder(args.device, config)
    text_processor = TextProcessor(config)
    model = FastSpeech2(config)

    if args.checkpoint != '':
        sd = torch.load(args.checkpoint, map_location=args.device)
        if 'model' in sd.keys():
            sd = sd['model']
    model.load_state_dict(sd)
    del sd  # to save mem
    model = model.to(args.device)
    torch.set_grad_enabled(False)

    try:
        lines = open(args.input).read().split('\n')
    except:
        print('Failed to open text file', args.input)
        print('Treating input as text')
        lines = [args.input]

    for line in lines:
        if len(line) == 0 or line.startswith('#'):
            continue
        logger.info(f'processing {line}')
        name, tokens = text_processor(line)
        tokens = tokens.to(args.device)
        seq_len = torch.tensor([tokens.shape[1]])
        tokens = tokens.unsqueeze(1)
        seq_len = seq_len.to(args.device)
        max_src_len = torch.max(seq_len)
        output = model(tokens, seq_len, max_src_len=max_src_len, d_control=args.duration)
        mel_pred, mel_postnet, d_pred, src_mask, mel_mask, mel_len = output

        # convert to waveform using vocoder
        mel_postnet = mel_postnet[0].transpose(0, 1).detach()
        mel_postnet += config['fbank']['mel_mean']
        wav = vocoder(mel_postnet)
        if config['synthesis']['normalize']:
            wav = normalize(wav)
        else:
            wav = to_int16(wav)
        dst_file = os.path.join(args.output_dir, f'{name}.wav')
        #np.save(dst_file+'.npy',mel_postnet.cpu().numpy())
        logger.info(f'writing file to {dst_file}')
        wavfile.write(dst_file, sr, wav)
