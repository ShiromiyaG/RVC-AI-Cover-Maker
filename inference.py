import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from yt_dlp import YoutubeDL
import click
import subprocess
import contextlib
import sys
from  glob import glob
from pydub import AudioSegment
from rvccli import run_infer_script
from musicsouceseparationtraining import proc_file
from shiromiyautils import ensemble_inputs
from shiromiyautils import add_audio_effects
from shiromiyautils import combine_audio
import torch
import audiofile as af
from uvr import models
from pathlib import Path
import gettext
import gdown
import requests
import zipfile
import json

gettext.bindtextdomain('RVCAIMaker', 'locale')
gettext.textdomain('RVCAIMaker')
_ = gettext.gettext

def get_last_modified_file(directory, filter=''):
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

@click.group()
def cli():
    pass

@click.command("download_yt")
@click.option('--link')
@click.option('--supress')
def download_yt(link, supress):
    options = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '256',
        }],
        'outtmpl': '/content/musicas/arquivos-originais/%(title)s.%(ext)s',
        'quiet': True
    }
    print(_("Downloading Youtube music..."))
    with supress_output(supress):
        with YoutubeDL(options) as ydl:
            ydl.download([link])
    print(_("Download of Youtube music complete!"))

@click.command("download_deezer")
@click.option('--link')
@click.option('--bf_secret')
@click.option('--track_url_key')
@click.option('--arl')
@click.option('--supress')
def download_deezer(link, bf_secret, track_url_key, arl, supress):
    print(_("Downloading Deezer music..."))
    with supress_output(supress):
        with open('/content/OrpheusDL/config/settings.json', 'r') as file:
            data = json.load(file)
        data['modules']['deezer']['bf_secret'] = bf_secret
        data['modules']['deezer']['track_url_key'] = track_url_key
        data['modules']['deezer']['arl'] = arl
        with open('/content/OrpheusDL/config/settings.json', 'w') as file:
            json.dump(data, file, indent=4)
        subprocess.run(["python", "OrpheusDL/orpheus.py", link])
    print(_("Download of Deezer complete!"))

@click.command("remove_backing_vocals_and_reverb")
@click.option('--input_file')
@click.option('--no_back_folder')
@click.option('--output_folder')
@click.option('--device')
@click.option('--supress')
def remove_backing_vocals_and_reverb(input_file, no_back_folder, output_folder, device, supress):
    with supress_output(supress):
        basename = os.path.basename(input_file).split(".")[0]
        # Conevert mp3 to flac
        if input_file.endswith(".mp3"):
            flac_filename = os.path.splitext(input_file)[0] + '.flac'
            if not os.path.exists(flac_filename):
                audio = AudioSegment.from_mp3(input_file)
                audio.export(f"{flac_filename}", format="flac")
                input_file = flac_filename

        Vr = models.VrNetwork(name="karokee_4band_v2_sn", other_metadata={'normaliz': False, 'aggressiveness': 0.05,'window_size': 320,'batch_size': 8,'is_tta': True},device=device, logger=None)
        with supress_output():
            res = Vr(input_file)
            vocals = res["vocals"]
            af.write(f"{no_back_folder}/{basename}_karokee_4band_v2_sn.wav", vocals, Vr.sample_rate)
        torch.cuda.empty_cache()
        filename_path = get_last_modified_file(no_back_folder)
        no_back_output = os.path.join(no_back_folder, filename_path)
    print(_(f"{basename} processing with karokee_4band_v2_sn is over!"))
    with supress_output(supress):
        # Reverb_HQ
        MDX = models.MDX(name="Reverb_HQ",  other_metadata={'segment_size': 384,'overlap': 0.75,'mdx_batch_size': 8,'semitone_shift': 0,'adjust': 1.08, 'denoise': False,'is_invert_spec': False,'is_match_frequency_pitch': True,'overlap_mdx': None},device=device, logger=None)
        with supress_output():
            res = MDX(no_back_output)
            no_reverb = res["no reverb"]
            af.write(f"{output_folder}/{basename}_Reverb_HQ.wav",  no_reverb, MDX.sample_rate)
        torch.cuda.empty_cache()
    print(_(f"{basename} processing with Reverb HQ is over!"))
    print(_("Vocal processing completed."))
    print(_("Separation complete!"))
    return [output for output in output_folder if "Reverb_HQ" in output]

@click.command("separate_vocals")
@click.option('--input_file')
@click.option('--vocal_ensemble')
@click.option('--algorithm_ensemble_vocals')
@click.option('--no_inst_folder')
@click.option('--no_back_folder')
@click.option('--output_folder')
@click.option('--device')
@click.option('--supress')
def separate_vocals(input_file, vocal_ensemble, algorithm_ensemble_vocals, no_inst_folder, no_back_folder, output_folder, device, supress):
    print(_("Separating vocals..."))
    with supress_output(supress):
        basename = os.path.basename(input_file).split(".")[0]
        # Conevert mp3 to flac
        if input_file.endswith(".mp3"):
            flac_filename = os.path.splitext(input_file)[0] + '.flac'
            if not os.path.exists(flac_filename):
                audio = AudioSegment.from_mp3(input_file)
                audio.export(f"{flac_filename}", format="flac")
                input_file = flac_filename
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
    print(_(f"{basename} processing with MDX23C-8KFFT-InstVoc_HQ is over!"))
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
        print(_(f"{basename} processing with BSRoformer is over!"))
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
        print(_("Processing of the first Ensemble is over!"))
    with supress_output(supress):
        # karokee_4band_v2_sn
        Vr = models.VrNetwork(name="karokee_4band_v2_sn", other_metadata={'normaliz': False, 'aggressiveness': 0.05,'window_size': 320,'batch_size': 8,'is_tta': True},device=device, logger=None)
        with supress_output():
            res = Vr(no_inst_output)
            vocals = res["vocals"]
            af.write(f"{no_back_folder}/{basename}_karokee_4band_v2_sn.wav", vocals, Vr.sample_rate)
        torch.cuda.empty_cache()
        no_back_output = get_last_modified_file(no_back_folder)
    print(_(f"{basename} processing with karokee_4band_v2_sn is over!"))
    with supress_output(supress):
        # Reverb_HQ
        MDX = models.MDX(name="Reverb_HQ",  other_metadata={'segment_size': 384,'overlap': 0.75,'mdx_batch_size': 8,'semitone_shift': 0,'adjust': 1.08, 'denoise': False,'is_invert_spec': False,'is_match_frequency_pitch': True,'overlap_mdx': None},device=device, logger=None)
        with supress_output():
            res = MDX(no_back_output)
            no_reverb = res["no reverb"]
            af.write(f"{output_folder}/{basename}_Reverb_HQ.wav",  no_reverb, MDX.sample_rate)
        torch.cuda.empty_cache()
    print(_(f"{basename} processing with Reverb HQ is over!"))
    print(_("Vocal processing completed."))
    print(_("Separation complete!"))
    return [output for output in output_folder if "Reverb_HQ" in output]

@click.command("separate_instrumentals")
@click.option('--input_file')
@click.option('--instrumental_ensemble')
@click.option('--algorithm_ensemble_inst')
@click.option('--stage1_dir')
@click.option('--stage2_dir')
@click.option('--final_output_dir')
@click.option('--device')
@click.option('--supress')
def separate_instrumentals(input_file, instrumental_ensemble, algorithm_ensemble_inst, stage1_dir, stage2_dir, final_output_dir, device, supress):
    print(_("Separating instrumentals..."))
    with supress_output(supress):
        basename = os.path.basename(input_file).split(".")[0]
        # Pass 1
        # Conevert mp3 to flac
        if input_file.endswith(".mp3"):
            flac_filename = os.path.splitext(input_file)[0] + '.flac'
            if not os.path.exists(flac_filename):
                audio = AudioSegment.from_mp3(input_file)
                audio.export(f"{flac_filename}", format="flac")
                os.remove(input_file)
                input_file = flac_filename
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
                    af.write(f"{stage1_dir}/{basename}_{model_name_without_ext}_(Instrumental).wav", instrumentals, Vr.sample_rate)
                    torch.cuda.empty_cache()
                    processed_models.append(model_name)
                print(_(f"{basename} processing with {model_name_without_ext} is over!"))
            if model_name == "UVR-MDX-NET-Inst_HQ_4.onnx":
                with supress_output(supress):
                    model_name_without_ext = model_name.split('.')[0]
                    MDX = models.MDX(name="UVR-MDX-NET-Inst_HQ_4", other_metadata={'segment_size': 256,'overlap': 0.75,'mdx_batch_size': 8,'semitone_shift': 0,'adjust': 1.08, 'denoise': False,'is_invert_spec': False,'is_match_frequency_pitch': True,'overlap_mdx': None},device=device, logger=None)
                    res = MDX(input_file)
                    instrumentals = res["instrumental"]
                    af.write(f"{stage1_dir}/{basename}_{model_name_without_ext}_(Instrumental).wav", instrumentals, MDX.sample_rate)
                    torch.cuda.empty_cache()
                    processed_models.append(model_name)
                print(_(f"{basename} processing with {model_name_without_ext} is over!"))
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
                print(_(f"{basename} processing with {model_name_without_ext} is over!"))
        models_names = [model for model in models_names if model not in processed_models]
    else:
        with supress_output(supress):
            model_name = "UVR-MDX-NET-Inst_HQ_4.onnx"
            model_name_without_ext = model_name.split('.')[0]
            MDX = models.MDX(name="UVR-MDX-NET-Inst_HQ_4.onnx", other_metadata={'segment_size': 256,'overlap': 0.75,'mdx_batch_size': 8,'semitone_shift': 0,'adjust': 1.08, 'denoise': False,'is_invert_spec': False,'is_match_frequency_pitch': True,'overlap_mdx': None},device=device, logger=None)
            res = MDX(input_file)
            instrumentals = res["instrumental"]
            af.write(f"{stage1_dir}/{basename}_Inst-HQ4.wav", instrumentals, MDX.sample_rate)
            torch.cuda.empty_cache()
            processed_models.append(model_name)
            final_output_path = os.path.join(stage1_dir, f"{basename}_{model_name_without_ext}.flac")
        print(_(f"{basename} processing with {model_name_without_ext} is over!"))
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
        print(_("Processing of the first Ensemble is over!"))
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
                    af.write(f"{stage2_dir}/{basename}_karokee_4band_v2_sn.wav", instrumentals, Vr.sample_rate)
                    torch.cuda.empty_cache()
                    processed_models.append(model_name)
                print(_(f"{basename} processing with {model_name_without_ext} is over!"))
            else:
                with supress_output(supress):
                    model_name_without_ext = model_name.split('.')[0]
                    output_path = os.path.join(stage2_dir, f"{basename}_{model_name_without_ext}_(Instrumental).wav")
                    pass2_outputs.append(output_path)
                    MDX = models.MDX(name=f"{model_name_without_ext}", other_metadata={'segment_size': 256,'overlap': 0.75,'mdx_batch_size': 8,'semitone_shift': 0,'adjust': 1.08, 'denoise': False,'is_invert_spec': False,'is_match_frequency_pitch': True,'overlap_mdx': None},device=device, logger=None)
                    res = MDX(input_file)
                    instrumentals = res["instrumental"]
                    af.write(f"{stage2_dir}/{basename}_{model_name}.wav", instrumentals, MDX.sample_rate)
                    torch.cuda.empty_cache()
                    processed_models.append(model_name)
                print(_(f"{basename} processing with {model_name_without_ext} is over!"))
            models_names = [model for model in models_names if model not in processed_models]

        # Third Ensemble
        with supress_output(supress):
            all_files = os.listdir(stage2_dir)
            pass2_outputs_filtered = [os.path.join(stage2_dir, output) for output in all_files if "Instrumental" in output]
            final_output_path = os.path.join(final_output_dir, f"{basename}_final_output.wav")
            ensemble_inputs(
                audio_input=pass2_outputs_filtered,
                algorithm=algorithm_ensemble_inst,
                is_normalization=False,
                wav_type_set="PCM_16",
                save_path=final_output_path,
                is_wave=False,
                is_array=False,
            )
        print(_("Processing of the second Ensemble is over!"))
    print(_("Instrumental processing completed."))
    if instrumental_ensemble == True:
        return [output for output in final_output_path if "instrumental" in output]
    else:
        return [output for output in stage1_dir if "instrumental" in output]

@click.command("rvc_ai")
@click.option('--input_path')
@click.option('--output_path')
@click.option('--rvc_model_name')
@click.option('--rvc_model_name_ext')
@click.option('--model_destination_folder')
@click.option('--rvc_model_link')
@click.option('--pitch')
@click.option('--filter_radius')
@click.option('--index_rate')
@click.option('--hop_length')
@click.option('--rms_mix_rate')
@click.option('--protect')
@click.option('--autotune')
@click.option('--f0method')
@click.option('--split_audio')
@click.option('--clean_audio')
@click.option('--clean_strength')
@click.option('--export_format')
@click.option('--supress')
def rvc_ai(input_path, output_path, rvc_model_name, rvc_model_name_ext, model_destination_folder, rvc_model_link, pitch, filter_radius, index_rate, hop_length, rms_mix_rate, protect, autotune, f0method, split_audio, clean_audio, clean_strength, export_format, supress):
    print("Downloading model...")
    with supress_output(supress):
        filename = rvc_model_name
        download_path = str(Path(model_destination_folder / filename)) + rvc_model_name_ext
        if "drive.google.com" in f"{rvc_model_link}":
            gdown.download(rvc_model_link, str(download_path), quiet=False)
        else:
            response = requests.get(rvc_model_link)
            with open(download_path, 'wb') as file:
                file.write(response.content)
        if str(download_path).endswith(".zip"):
            Path(f'/content/RVC_CLI/logs/{rvc_model_name}').mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(f"/content/RVC_CLI/logs/{rvc_model_name}"+".zip")
    print("Download complete.")
    with supress_output(supress):
        current_dir = "/content/RVC_CLI"
        model_folder = os.path.join(current_dir, f"logs/{rvc_model_name}")

        if not os.path.exists(model_folder):
            raise FileNotFoundError(f"Model directory not found: {model_folder}")

        files_in_folder = os.listdir(f"{model_destination_folder}/{rvc_model_name}")
        pth_file = find_files(model_destination_folder, ".pth")
        index_file = find_files(model_destination_folder, ".index")

        if pth_file is None or index_file is None:
            raise FileNotFoundError("No model found.")

        output_path = "/content/output_rvc.flac"
        export_format = "FLAC"
        run_infer_script(
            f0up_key=pitch,
            filter_radius=filter_radius,
            index_rate=index_rate,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            hop_length=hop_length,
            f0method=f0method,
            input_path=input_path,
            output_path=output_path,
            pth_path=pth_file,
            index_path=index_file,
            split_audio=split_audio,
            f0autotune=autotune,
            clean_audio=clean_audio,
            clean_strength=clean_strength,
            export_format=export_format,
        )
    print(_("RVC AI processing complete!"))
    return output_path.replace(".flac", f".{export_format.lower()}")

@click.command("reverb")
@click.option('--audio_path')
@click.option('--reverb_size')
@click.option('--reverb_wetness')
@click.option('--reverb_dryness')
@click.option('--reverb_damping')
@click.option('--output_path')
@click.option('--supress')
def reverb(audio_path, reverb_size, reverb_wetness, reverb_dryness, reverb_damping, output_path, supress):
    with supress_output(supress):
        reverb_size = float(reverb_size)
        reverb_dry = float(reverb_dry)
        reverb_wet = float(reverb_wet)
        reverb_damping = float(reverb_damping)
        add_audio_effects(
            audio_path=audio_path,
            reverb_size=reverb_size,
            reverb_wet=reverb_wetness,
            reverb_dry=reverb_dryness,
            reverb_damping=reverb_damping,
            output_path=output_path,
        )
    return

@click.command("remove_noise")
@click.option('--audio_path')
@click.option('--noise_db_limit')
@click.option('--output_path')
@click.option('--supress')
def remove_noise(noise_db_limit, audio_path, output_path, supress):
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

@click.command("mix_audio")
@click.option('--audio_paths')
@click.option('--output_path')
@click.option('--main_gain')
@click.option('--inst_gain')
@click.option('--output_format')
@click.option('--supress')
def mix_audio(audio_paths, output_path, main_gain, inst_gain, output_format, supress):
    with supress_output(supress):
        combine_audio(
            audio_paths=audio_paths,
            output_path=output_path,
            main_gain=main_gain,
            inst_gain=inst_gain,
            output_format=output_format,
            supress=supress,
        )

@click.command("ensemble")
@click.option('--input_folder')
@click.option('--algorithm_ensemble')
@click.option('--output_path')
@click.option('--supress')
def ensemble(input_folder, algorithm_ensemble, output_path, supress):
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

def main():
    cli.add_command(download_yt)
    cli.add_command(download_deezer)
    cli.add_command(remove_backing_vocals_and_reverb)
    cli.add_command(separate_vocals)
    cli.add_command(separate_instrumentals)
    cli.add_command(rvc_ai)
    cli.add_command(reverb)
    cli.add_command(mix_audio)
    cli.add_command(ensemble)
    cli()

if __name__ == "__main__":
    main()
