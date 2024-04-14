import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from yt_dlp import YoutubeDL
import subprocess
import contextlib
import sys
from glob import glob
from pydub import AudioSegment
from musicsouceseparationtraining import proc_file
from shiromiyautils import ensemble_inputs
from shiromiyautils import add_audio_effects
from shiromiyautils import combine_audio
import torch
import audiofile as af
from uvr import models
from pathlib import Path
import gdown
import requests
import zipfile
import json
import argparse

def get_last_modified_file(directory, filter=''):
    if isinstance(directory, set):
        directory = list(directory)[0]
    arquivos = glob(directory + "/*")
    if filter != '':
        arquivos = [arquivo for arquivo in arquivos if filter in arquivo]
    if arquivos:
        return max(arquivos, key=os.path.getmtime)
    else:
        return None

def find_files(directory, extensions):
  files = glob(f'{directory}/**/*{extensions}', recursive=True)
  return files[0]

@contextlib.contextmanager
def supress_output(supress=True):
    if supress:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    else:
        yield

def download_yt(link, supress, language=None):
    options = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'flac',
            'preferredquality': '256',
        }],
        'outtmpl': '/content/musicas/arquivos-originais/%(title)s.%(ext)s',
        'quiet': True
    }
    if language == "BR":
        print("Fazendo download da música do Deezer...")
    else:
        print("Downloading Youtube music...")
    with supress_output(supress):
        with YoutubeDL(options) as ydl:
            ydl.download([link])
    if language == "BR":
        print("Download da música do Deezer completo!")
    else:
        print("Download of Youtube music complete!")

def download_deezer(link, bf_secret, track_url_key, arl, supress, language=None):
    if language == "BR":
        print("Fazendo download da música do Deezer...")
    else:
        print("Downloading Deezer music...")
    with supress_output(supress):
        with open('/content/OrpheusDL/config/settings.json', 'r') as file:
            data = json.load(file)
        data['modules']['deezer']['bf_secret'] = bf_secret
        data['modules']['deezer']['track_url_key'] = track_url_key
        data['modules']['deezer']['arl'] = arl
        with open('/content/OrpheusDL/config/settings.json', 'w') as file:
            json.dump(data, file, indent=4)
        subprocess.run(["python", "OrpheusDL/orpheus.py", link])
    if language == "BR":
        print("Download da música do Deezer completo!")
    else:
        print("Download of Deezer complete!")

def remove_backing_vocals_and_reverb(input_file, no_back_folder, output_folder, device, supress, language=None):
    with supress_output(supress):
        basename = os.path.basename(input_file).split(".")[0]
        Vr = models.VrNetwork(name="karokee_4band_v2_sn", other_metadata={'normaliz': False, 'aggressiveness': 0.05,'window_size': 320,'batch_size': 8,'is_tta': True},device=device, logger=None)
        with supress_output():
            res = Vr(input_file)
            vocals = res["vocals"]
            af.write(f"{no_back_folder}/{basename}_karokee_4band_v2_sn.wav", vocals, Vr.sample_rate)
        torch.cuda.empty_cache()
        filename_path = get_last_modified_file(no_back_folder)
        no_back_output = os.path.join(no_back_folder, filename_path)
    if language == "BR":
        print(f"{basename} processamento com karokee_4band_v2_sn finalizado!")
    else:
        print(f"{basename} processing with karokee_4band_v2_sn is over!")
    with supress_output(supress):
        # Reverb_HQ
        MDX = models.MDX(name="Reverb_HQ",  other_metadata={'segment_size': 384,'overlap': 0.75,'mdx_batch_size': 8,'semitone_shift': 0,'adjust': 1.08, 'denoise': False,'is_invert_spec': False,'is_match_frequency_pitch': True,'overlap_mdx': None},device=device, logger=None)
        with supress_output():
            res = MDX(no_back_output)
            no_reverb = res["no reverb"]
            af.write(f"{output_folder}/{basename}_Reverb_HQ.wav",  no_reverb, MDX.sample_rate)
        torch.cuda.empty_cache()
    if language == "BR":
        print(f"{basename} processamento com Reverb HQ finalizado!")
        print("Processamento de vocal completo!")
    else:
        print(f"{basename} processing with Reverb HQ is over!")
        print("Vocal processing completed.")
    return [output for output in output_folder if "Reverb_HQ" in output]

def separate_vocals(input_file, vocal_ensemble, algorithm_ensemble_vocals, no_inst_folder, no_back_folder, output_folder, device, supress, language=None):
    if language == "BR":
        print("Separando vocais...")
    else:
        print("Separating vocals...")
    with supress_output(supress):
        basename = os.path.basename(input_file).split(".")[0]
        # MDX23C-8KFFT-InstVoc_HQ
        MDX23C_args = [
            "--model_type", "mdx23c",
            "--config_path", "Music_Source_Separation_Training/models/model_2_stem_full_band_8k.yaml",
            "--start_check_point", "Music_Source_Separation_Training/models/MDX23C-8KFFT-InstVoc_HQ.ckpt",
            "--input_file", f"{input_file}",
            "--store_dir", f"{no_inst_folder}",
        ]
        proc_file(MDX23C_args)
        file = get_last_modified_file(no_inst_folder, "Vocals")
        base_name = Path(file).stem
        extension = Path(file).suffix
        new_name = f"{base_name}MDX23C-8KFFT-InstVoc_HQ{extension}"
        os.rename(file, os.path.join(os.path.dirname(file), new_name))
    if language == "BR":
        print(f"{basename} processamento com MDX23C-8KFFT-InstVoc_HQ finalizado!")
    else:
        print(f"{basename} processing with MDX23C-8KFFT-InstVoc_HQ is over!")
    # Ensemble Vocals
    if vocal_ensemble:
        with supress_output(supress):
            lista = []
            lista.append(get_last_modified_file(no_inst_folder, "Vocals"))
            BSRoformer_args = [
                "--model_type", "bs_roformer",
                "--config_path", "Music_Source_Separation_Training/models/model_bs_roformer_ep_317_sdr_12.9755.yaml",
                "--start_check_point", "Music_Source_Separation_Training/models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                "--input_file", f"{input_file}",
                "--store_dir", f"{no_inst_folder}",
            ]
            proc_file(BSRoformer_args)
        if language == "BR":
            print(f"{basename} processamento com BSRoformer finalizado!")
        else:
            print(f"{basename} processing with BSRoformer is over!")
        with supress_output(supress):
            lista.append(get_last_modified_file(no_inst_folder, "Vocals"))
            ensemble_voc = os.path.join(no_inst_folder, f"{basename}_ensemble1.wav")
            ensemble_inputs(
                audio_input=lista,
                algorithm=algorithm_ensemble_vocals,
                is_normalization=False,
                wav_type_set="PCM_16",
                save_path=ensemble_voc,
                is_wave=False,
                is_array=False,
            )
            no_inst_output = ensemble_voc
        if language == "BR":
            print("Processamento do primeiro Ensemble finalizado!")
        else:
            print("Processing of the first Ensemble is over!")
    with supress_output(supress):
        # karokee_4band_v2_sn
        Vr = models.VrNetwork(name="karokee_4band_v2_sn", other_metadata={'normaliz': False, 'aggressiveness': 0.05,'window_size': 320,'batch_size': 8,'is_tta': True},device=device, logger=None)
        with supress_output():
            res = Vr(no_inst_output)
            vocals = res["vocals"]
            af.write(f"{no_back_folder}/{basename}_karokee_4band_v2_sn.wav", vocals, Vr.sample_rate)
        torch.cuda.empty_cache()
        no_back_output = get_last_modified_file(no_back_folder)
    if language == "BR":
        print(f"{basename} processamento com karokee_4band_v2_sn finalizado!")
    else:
        print(f"{basename} processing with karokee_4band_v2_sn is over!")
    with supress_output(supress):
        # Reverb_HQ
        MDX = models.MDX(name="Reverb_HQ",  other_metadata={'segment_size': 384,'overlap': 0.75,'mdx_batch_size': 8,'semitone_shift': 0,'adjust': 1.08, 'denoise': False,'is_invert_spec': False,'is_match_frequency_pitch': True,'overlap_mdx': None},device=device, logger=None)
        with supress_output():
            res = MDX(no_back_output)
            no_reverb = res["no reverb"]
            af.write(f"{output_folder}/{basename}_Reverb_HQ.wav",  no_reverb, MDX.sample_rate)
        torch.cuda.empty_cache()
    if language == "BR":
        print(f"{basename} processamento com Reverb HQ finalizado!")
        print("Processamento de vocal completo!")
    else:
        print(f"{basename} processing with Reverb HQ is over!")
        print("Vocal processing completed.")
    return get_last_modified_file(output_folder)

def separate_instrumentals(input_file, instrumental_ensemble, algorithm_ensemble_inst, stage1_dir, stage2_dir, final_output_dir, device, supress, language=None):
    if language == "BR":
        print("Separando instrumentais...")
    else:
        print("Separating instrumentals...")
    basename = os.path.basename(input_file).split(".")[0]
    # Pass 1
    if instrumental_ensemble:
        processed_models = []
        models_names = ["5_HP-Karaoke-UVR.pth", "UVR-MDX-NET-Inst_HQ_4.onnx", "htdemucs.yaml"]
        for model_name in models_names:
            if model_name == "5_HP-Karaoke-UVR.pth":
                with supress_output(supress):
                    model_name_without_ext = model_name.split('.')[0]
                    Vr = models.VrNetwork(name="5_HP-Karaoke-UVR", other_metadata={'normaliz': False, 'aggressiveness': 0.05,'window_size': 320,'batch_size': 8,'is_tta': True},device=device, logger=None)
                    res = Vr(input_file)
                    instrumentals = res["instrumental"]
                    af.write(f"{stage1_dir}/{basename}_{model_name_without_ext}Instrumental).wav", instrumentals, Vr.sample_rate)
                    torch.cuda.empty_cache()
                    processed_models.append(model_name)
                print(f"{basename} processing with {model_name_without_ext} is over!")
            if model_name == "UVR-MDX-NET-Inst_HQ_4.onnx":
                with supress_output(supress):
                    model_name_without_ext = model_name.split('.')[0]
                    MDX = models.MDX(name="UVR-MDX-NET-Inst_HQ_4", other_metadata={'segment_size': 256,'overlap': 0.75,'mdx_batch_size': 8,'semitone_shift': 0,'adjust': 1.08, 'denoise': False,'is_invert_spec': False,'is_match_frequency_pitch': True,'overlap_mdx': None},device=device, logger=None)
                    res = MDX(input_file)
                    instrumentals = res["instrumental"]
                    af.write(f"{stage1_dir}/{basename}_{model_name_without_ext}Instrumental).wav", instrumentals, MDX.sample_rate)
                    torch.cuda.empty_cache()
                    processed_models.append(model_name)
                print(f"{basename} processing with {model_name_without_ext} is over!")
            if model_name == "htdemucs.yaml":
                with supress_output(supress):
                    model_name_without_ext = model_name.split('.')[0]
                    demucs = models.Demucs(name="htdemucs",other_metadata={"segment":2, "split":True},device=device, logger=None)
                    res = demucs(input_file)
                    drum = res["drums"]
                    bass = res["bass"]
                    other = res["other"]
                    af.write(f"{stage1_dir}/{basename}_(Drums)_htdemucs.wav", drum, demucs.sample_rate)
                    af.write(f"{stage1_dir}/{basename}_(Bass)_htdemucs.wav", bass, demucs.sample_rate)
                    af.write(f"{stage1_dir}/{basename}_(Other)_htdemucs.wav", other, demucs.sample_rate)
                    torch.cuda.empty_cache()
                    audio_files = [
                        os.path.join(stage1_dir, f"{basename}_(Drums)_htdemucs.wav"),
                        os.path.join(stage1_dir, f"{basename}_(Bass)_htdemucs.wav"),
                        os.path.join(stage1_dir, f"{basename}_(Other)_htdemucs.wav")
                    ]

                    combined_audio = AudioSegment.from_file(audio_files[0], format="flac")

                    for audio_file in audio_files[1:]:
                        audio = AudioSegment.from_file(audio_file, format="flac")
                        combined_audio = combined_audio.overlay(audio)

                    combined_audio.export(f"{stage1_dir}/{basename}_demucs_(Instrumental).flac", format="flac")

                    for audio_file in audio_files:
                        os.remove(audio_file)
                    processed_models.append(model_name)
                if language == "BR":
                    print(f"{basename} processamento com {model_name_without_ext} finalizado!")
                else:
                    print(f"{basename} processing with {model_name_without_ext} is over!")
        models_names = [model for model in models_names if model not in processed_models]
    else:
        with supress_output(supress):
            model_name = "UVR-MDX-NET-Inst_HQ_4.onnx"
            model_name_without_ext = model_name.split('.')[0]
            MDX = models.MDX(name="UVR-MDX-NET-Inst_HQ_4", other_metadata={'segment_size': 256,'overlap': 0.75,'mdx_batch_size': 8,'semitone_shift': 0,'adjust': 1.08, 'denoise': False,'is_invert_spec': False,'is_match_frequency_pitch': True,'overlap_mdx': None},device=device, logger=None)
            res = MDX(input_file)
            instrumentals = res["instrumental"]
            af.write(f"{stage1_dir}/{basename}_Inst-HQ4_(instrumental).wav", instrumentals, MDX.sample_rate)
            torch.cuda.empty_cache()
            processed_models.append(model_name)
            final_output_path = os.path.join(stage1_dir, f"{basename}_{model_name_without_ext}.flac")
        if language == "BR":
            print(f"{basename} processamento com {model_name_without_ext} finalizado!")
        else:
            print(f"{basename} processing with {model_name_without_ext} is over!")
    # Second Ensemble
    if instrumental_ensemble:
        with supress_output(supress):
            all_files = os.listdir(stage1_dir)
            pass1_outputs_filtered = [os.path.join(stage1_dir, output) for output in all_files if "Instrumental" in output]
            ensemble1_output = os.path.join(stage1_dir, f"{basename}_ensemble1.wav")
            ensemble_inputs(
                audio_input=pass1_outputs_filtered,
                algorithm=algorithm_ensemble_inst,
                is_normalization=False,
                wav_type_set="PCM_16",
                save_path=ensemble1_output,
                is_wave=False,
                is_array=False,
            )
        if language == "BR":
            print("Processamento do primeiro Ensemble finalizado!")
        else:
            print("Processing of the first Ensemble is over!")
        # Pass 2
        processed_models = []
        pass2_outputs = []
        models_names = ["karokee_4band_v2_sn.pth", "UVR-MDX-NET-Inst_HQ_4.onnx", "Kim_Vocal_2.onnx"]
        for model_name in models_names:
            if model_name == "karokee_4band_v2_sn.pth":
                with supress_output(supress):
                    model_name_without_ext = model_name.split('.')[0]
                    output_path = os.path.join(stage2_dir, f"{basename}_{model_name_without_ext}_(Instrumental).wav")
                    pass2_outputs.append(output_path)
                    Vr = models.VrNetwork(name="karokee_4band_v2_sn", other_metadata={'aggressiveness': 0.05,'window_size': 320,'batch_size': 8,'is_tta': True},device=device, logger=None)
                    res = Vr(ensemble1_output)
                    instrumentals = res["instrumental"]
                    af.write(f"{stage2_dir}/{basename}_karokee_4band_v2_sn_(Instrumental).wav", instrumentals, Vr.sample_rate)
                    torch.cuda.empty_cache()
                    processed_models.append(model_name)
                if language == "BR":
                    print(f"{basename} processamento com {model_name_without_ext} finalizado!")
                else:
                    print(f"{basename} processing with {model_name_without_ext} is over!")
            else:
                with supress_output(supress):
                    model_name_without_ext = model_name.split('.')[0]
                    output_path = os.path.join(stage2_dir, f"{basename}_{model_name_without_ext}_(Instrumental).wav")
                    pass2_outputs.append(output_path)
                    MDX = models.MDX(name=f"{model_name_without_ext}", other_metadata={'segment_size': 256,'overlap': 0.75,'mdx_batch_size': 8,'semitone_shift': 0,'adjust': 1.08, 'denoise': False,'is_invert_spec': False,'is_match_frequency_pitch': True,'overlap_mdx': None},device=device, logger=None)
                    res = MDX(input_file)
                    instrumentals = res["instrumental"]
                    af.write(f"{stage2_dir}/{basename}_{model_name}_(Instrumental).wav", instrumentals, MDX.sample_rate)
                    torch.cuda.empty_cache()
                    processed_models.append(model_name)
                if language == "BR":
                    print(f"{basename} processamento com {model_name_without_ext} finalizado!")
                else:
                    print(f"{basename} processing with {model_name_without_ext} is over!")
            models_names = [model for model in models_names if model not in processed_models]

        # Third Ensemble
        with supress_output(supress):
            all_files = os.listdir(stage2_dir)
            pass2_outputs_filtered = [os.path.join(stage2_dir, output) for output in all_files if "Instrumental" in output]
            final_output_path = os.path.join(final_output_dir, f"{basename}_final_output_(instrumental).wav")
            ensemble_inputs(
                audio_input=pass2_outputs_filtered,
                algorithm=algorithm_ensemble_inst,
                is_normalization=False,
                wav_type_set="PCM_16",
                save_path=final_output_path,
                is_wave=False,
                is_array=False,
            )
        if language == "BR":
            print("Processamento do segundo Ensemble finalizado!")
            print("Processamento de instrumentais completo.")
        else:
            print("Processing of the second Ensemble is over!")
            print("Instrumental processing completed.")
    if instrumental_ensemble == True:
        get_last_modified_file(final_output_dir)
    else:
        get_last_modified_file(stage1_dir)

def rvc_ai(rvc_model_name, rvc_model_name_ext, model_destination_folder, rvc_model_link, supress, language=None):
    if language == "BR":
        print("Processando com RVC AI...")
        print("Baixando modelo...")
    else:
        print("Processing with RVC AI...")
        print("Downloading model...")
    with supress_output(supress):
        filename = rvc_model_name
        download_path = os.path.join(f"{model_destination_folder}/" + filename + rvc_model_name_ext)
        if "drive.google.com" in f"{rvc_model_link}":
            gdown.download(rvc_model_link, str(download_path), quiet=False)
        else:
            response = requests.get(rvc_model_link)
            with open(download_path, 'wb') as file:
                file.write(response.content)
        if str(download_path).endswith(".zip"):
            extraction_folder = os.path.join("/content/RVC_CLI-1.1.1/logs", rvc_model_name)
            Path(extraction_folder).mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(extraction_folder)
    if language == "BR":
        print("Download do modelo completo.")
    else:
        print("Download complete.")

def reverb(audio_path, reverb, reverb_size, reverb_wetness, reverb_dryness, reverb_damping, reverb_width, limiter, limiter_threshold_db, limiter_release_time, limiter_ceiling_db, compressor, compressor_ratio, compressor_threshold_db, compressor_attack_ms, compressor_release_ms, compressor_knee_db, compressor_makeup_gain_db, output_path, supress, language=None):
    with supress_output(supress):
        add_audio_effects(
            audio_path=audio_path,
            reverb=reverb,
            reverb_size=float(reverb_size),
            reverb_wet=float(reverb_wetness),
            reverb_dry=float(reverb_dryness),
            reverb_damping=float(reverb_damping),
            reverb_width=float(reverb_width),
            limiter=limiter,
            limiter_threshold_db=float(limiter_threshold_db),
            limiter_release_time=float(limiter_release_time),
            limiter_ceiling_db=float(limiter_ceiling_db),
            compressor=compressor,
            compressor_ratio=float(compressor_ratio),
            compressor_threshold_db=float(compressor_threshold_db),
            compressor_attack_ms=float(compressor_attack_ms),
            compressor_release_ms=float(compressor_release_ms),
            compressor_knee_db=float(compressor_knee_db),
            compressor_makeup_gain_db=float(compressor_makeup_gain_db),
            output_path=output_path,
        )
    return

def remove_noise(noise_db_limit, audio_path, output_path, supress, language=None):
    with supress_output(supress):
        audio = AudioSegment.from_file(audio_path)
        db_limit = noise_db_limit
        silenced_audio = AudioSegment.silent(duration=len(audio))
        chunk_length = 100
        for i in range(0, len(audio), chunk_length):
            chunk = audio[i:i+chunk_length]
            if chunk.dBFS > db_limit:
                silenced_audio = silenced_audio.overlay(chunk, position=i)
        silenced_audio.export(output_path, format="wav")
    return output_path

def mix_audio(input_file, vocals_path, inst_path, output_path, main_gain, inst_gain, output_format, rvc_model_name, supress, language=None):
    with supress_output(supress):
        output_path = f"{output_path}/{input_file}_({rvc_model_name} Version).{output_format}"
        main_vocal_audio = AudioSegment.from_file(vocals_path, format='flac') + float(main_gain)
        instrumental_audio = AudioSegment.from_file(inst_path, format='flac') + float(inst_gain)
        main_vocal_audio.overlay(instrumental_audio).export(output_path, format=output_format)
        return output_path

def ensemble(input_folder, algorithm_ensemble, output_path, supress, language=None):
    with supress_output(supress):
        files = [file for file in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, file))]
        ensemble_inputs(
            audio_input=files,
            algorithm=algorithm_ensemble,
            is_normalization=False,
            wav_type_set="PCM_16",
            save_path=output_path,
            is_wave=False,
            is_array=False,
        )

def cpu_mode(input_file, vocal_ensemble, algorithm_ensemble_vocals, no_inst_folder, no_back_folder, output_vocals, output_instrumentals, device, supress, language=None):
    if language == "BR":
        print("Separando vocais...")
    else:
        print("Separating vocals...")
    with supress_output(supress):
        basename = os.path.basename(input_file).split(".")[0]
        # Voc_FT
        MDX = models.MDX(name="Voc_FT",  other_metadata={'segment_size': 256,'overlap': 0.25,'mdx_batch_size': 10,'semitone_shift': 0,'adjust': 1.08, 'denoise': False,'is_invert_spec': False,'is_match_frequency_pitch': True,'overlap_mdx': None},device=device, logger=None)
        res = MDX(input_file)
        vocals = res["vocals"]
        af.write(f"{no_inst_folder}/{basename}_Voc_FT_(Vocals).wav",  vocals, MDX.sample_rate)
    if language == "BR":
        print(f"{basename} processamento com Voc_FT finalizado!")
    else:
        print(f"{basename} processing with Voc_FT is over!")
    with supress_output(supress):
        # karokee_4band_v2_sn
        MDX = models.MDX(name="UVR_MDXNET_KARA_2", other_metadata={'segment_size': 256,'overlap': 0.25,'mdx_batch_size': 10,'semitone_shift': 0,'adjust': 1.08, 'denoise': False,'is_invert_spec': False,'is_match_frequency_pitch': True,'overlap_mdx': None},device=device, logger=None)
        res = MDX(no_inst_folder)
        vocals = res["vocals"]
        af.write(f"{no_back_folder}/{basename}_UVR_MDXNET_KARA_2.wav", vocals, Vr.sample_rate)
        no_back_output = get_last_modified_file(no_back_folder)
    if language == "BR":
        print(f"{basename} processamento com UVR_MDXNET_KARA_2 finalizado!")
    else:
        print(f"{basename} processing with UVR_MDXNET_KARA_2 is over!")
    with supress_output(supress):
        # Reverb_HQ
        MDX = models.MDX(name="Reverb_HQ",  other_metadata={'segment_size': 256,'overlap': 0.25,'mdx_batch_size': 8,'semitone_shift': 0,'adjust': 1.08, 'denoise': False,'is_invert_spec': False,'is_match_frequency_pitch': True,'overlap_mdx': None},device=device, logger=None)
        res = MDX(no_back_output)
        no_reverb = res["no reverb"]
        af.write(f"{output_vocals}/{basename}_Reverb_HQ.wav",  no_reverb, MDX.sample_rate)
    if language == "BR":
        print(f"{basename} processamento com Reverb HQ finalizado!")
        print("Processamento de vocal completo!")
    else:
        print(f"{basename} processing with Reverb HQ is over!")
        print("Vocal processing completed.")
    output = []
    output.append(get_last_modified_file(output_vocals))
    with supress_output(supress):
        # Reverb_HQ
        MDX = models.MDX(name="UVR-MDX-NET-Inst_HQ_4",  other_metadata={'segment_size': 256,'overlap': 0.25,'mdx_batch_size': 8,'semitone_shift': 0,'adjust': 1.08, 'denoise': False,'is_invert_spec': False,'is_match_frequency_pitch': True,'overlap_mdx': None},device=device, logger=None)
        res = MDX(no_back_output)
        inst = res["instrumental"]
        af.write(f"{output_instrumentals}/{basename}_Reverb_HQ.wav",  inst, MDX.sample_rate)
    if language == "BR":
        print(f"{basename} processamento com Inst_HQ_4 finalizado!")
    else:
        print(f"{basename} processing with Inst_HQ_4 is over!")
    output.append(get_last_modified_file(output_instrumentals))
    return output

def main():
    parser = argparse.ArgumentParser(description="CLI")
    subparsers = parser.add_subparsers()

    download_yt_parser = subparsers.add_parser('download_yt')
    download_yt_parser.add_argument('--link')
    download_yt_parser.add_argument('--supress')
    download_yt_parser.add_argument('--language')
    download_yt_parser.set_defaults(func=download_yt)

    deezer_parser = subparsers.add_parser('download_deezer')
    deezer_parser.add_argument('--link')
    deezer_parser.add_argument('--bf_secret')
    deezer_parser.add_argument('--track_url_key')
    deezer_parser.add_argument('--arl')
    deezer_parser.add_argument('--supress')
    deezer_parser.add_argument('--language')
    deezer_parser.set_defaults(func=download_deezer)

    rbvr_parser = subparsers.add_parser('remove_backing_vocals_and_reverb')
    rbvr_parser.add_argument('--input_file')
    rbvr_parser.add_argument('--no_back_folder')
    rbvr_parser.add_argument('--output_folder')
    rbvr_parser.add_argument('--device')
    rbvr_parser.add_argument('--supress')
    rbvr_parser.add_argument('--language')
    rbvr_parser.set_defaults(func=remove_backing_vocals_and_reverb)

    separate_vocals_parser = subparsers.add_parser('separate_vocals')
    separate_vocals_parser.add_argument('--input_file')
    separate_vocals_parser.add_argument('--vocal_ensemble')
    separate_vocals_parser.add_argument('--algorithm_ensemble_vocals')
    separate_vocals_parser.add_argument('--no_inst_folder')
    separate_vocals_parser.add_argument('--no_back_folder')
    separate_vocals_parser.add_argument('--output_folder')
    separate_vocals_parser.add_argument('--device')
    separate_vocals_parser.add_argument('--supress')
    separate_vocals_parser.add_argument('--language')
    separate_vocals_parser.set_defaults(func=separate_vocals)

    separate_instrumentals_parser = subparsers.add_parser('separate_instrumentals')
    separate_instrumentals_parser.add_argument('--input_file')
    separate_instrumentals_parser.add_argument('--instrumental_ensemble')
    separate_instrumentals_parser.add_argument('--algorithm_ensemble_inst')
    separate_instrumentals_parser.add_argument('--stage1_dir')
    separate_instrumentals_parser.add_argument('--stage2_dir')
    separate_instrumentals_parser.add_argument('--final_output_dir')
    separate_instrumentals_parser.add_argument('--device')
    separate_instrumentals_parser.add_argument('--supress')
    separate_instrumentals_parser.add_argument('--language')
    separate_instrumentals_parser.set_defaults(func=separate_instrumentals)

    rvc_parser = subparsers.add_parser("rvc_ai")
    rvc_parser.add_argument('--rvc_model_name')
    rvc_parser.add_argument('--rvc_model_name_ext')
    rvc_parser.add_argument('--model_destination_folder')
    rvc_parser.add_argument('--rvc_model_link')
    rvc_parser.add_argument('--supress')
    rvc_parser.add_argument('--language')
    rvc_parser.set_defaults(func=rvc_ai)

    reverb_parser = subparsers.add_parser('reverb')
    reverb_parser.add_argument('--audio_path')
    reverb_parser.add_argument('--reverb')
    reverb_parser.add_argument('--reverb_size', type=float)
    reverb_parser.add_argument('--reverb_wetness', type=float)
    reverb_parser.add_argument('--reverb_dryness', type=float)
    reverb_parser.add_argument('--reverb_damping', type=float)
    reverb_parser.add_argument('--reverb_width', type=float)
    reverb_parser.add_argument('--limiter')
    reverb_parser.add_argument('--limiter_threshold_db', type=float)
    reverb_parser.add_argument('--limiter_release_time', type=float)
    reverb_parser.add_argument('--limiter_ceiling_db', type=float)
    reverb_parser.add_argument('--compressor')
    reverb_parser.add_argument('--compressor_ratio', type=float)
    reverb_parser.add_argument('--compressor_threshold_db', type=float)
    reverb_parser.add_argument('--compressor_attack_ms', type=float)
    reverb_parser.add_argument('--compressor_release_ms', type=float)
    reverb_parser.add_argument('--compressor_knee_db', type=float)
    reverb_parser.add_argument('--compressor_makeup_gain_db', type=float)
    reverb_parser.add_argument('--output_path')
    reverb_parser.add_argument('--supress')
    reverb_parser.add_argument('--language')
    reverb_parser.set_defaults(func=reverb)

    remove_noise_parser = subparsers.add_parser('remove_noise')
    remove_noise_parser.add_argument('--audio_path')
    remove_noise_parser.add_argument('--noise_db_limit', type=float)
    remove_noise_parser.add_argument('--output_path')
    remove_noise_parser.add_argument('--supress')
    remove_noise_parser.add_argument('--language')
    remove_noise_parser.set_defaults(func=remove_noise)

    mix_audio_parser = subparsers.add_parser('mix_audio')
    mix_audio_parser.add_argument('--input_file')
    mix_audio_parser.add_argument('--vocals_path')
    mix_audio_parser.add_argument('--inst_path')
    mix_audio_parser.add_argument('--output_path')
    mix_audio_parser.add_argument('--main_gain', type=float)
    mix_audio_parser.add_argument('--inst_gain', type=float)
    mix_audio_parser.add_argument('--output_format')
    mix_audio_parser.add_argument('--rvc_model_name')
    mix_audio_parser.add_argument('--supress')
    mix_audio_parser.add_argument('--language')
    mix_audio_parser.set_defaults(func=mix_audio)

    ensemble_parser = subparsers.add_parser('ensemble')
    ensemble_parser.add_argument('--input_folder')
    ensemble_parser.add_argument('--algorithm_ensemble')
    ensemble_parser.add_argument('--output_path')
    ensemble_parser.add_argument('--supress')
    ensemble_parser.add_argument('--language')
    ensemble_parser.set_defaults(func=ensemble)

    cpu_mode_parser = subparsers.add_parser('cpu_mode')
    cpu_mode_parser.add_argument('--input_file')
    cpu_mode_parser.add_argument('--vocal_ensemble')
    cpu_mode_parser.add_argument('--algorithm_ensemble_vocals')
    cpu_mode_parser.add_argument('--no_inst_folder')
    cpu_mode_parser.add_argument('--no_back_folder')
    cpu_mode_parser.add_argument('--output_vocals')
    cpu_mode_parser.add_argument('--output_instrumentals')
    cpu_mode_parser.add_argument('--device')
    cpu_mode_parser.add_argument('--supress')
    cpu_mode_parser.add_argument('--language')
    cpu_mode_parser.set_defaults(func=cpu_mode)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
