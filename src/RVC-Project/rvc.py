from pathlib import Path
import argparse
from dotenv import load_dotenv
from scipy.io import wavfile

from rvc.modules.vc.modules import VC

parser = argparse.ArgumentParser(description='Voice Conversion')

parser.add_argument('--pth_file', type=str, required=True, help='Path to the .pth file')
parser.add_argument('--input_path', type=str, required=True, help='Path to the input audio file')
parser.add_argument('--pitch', type=int, required=True, help='Pitch')
parser.add_argument('--f0method', type=str, required=True, help='F0 method')
parser.add_argument('--filter_radius', type=int, required=True, help='Filter radius')
parser.add_argument('--index_rate', type=float, required=True, help='Index rate')
parser.add_argument('--rms_mix_rate', type=float, required=True, help='RMS mix rate')
parser.add_argument('--protect', type=float, required=True, help='Protect')
parser.add_argument('--output_file', type=str, required=True, help='output file')

args = parser.parse_args()

def main():
      vc = VC()
      vc.get_vc(args.pth_file)
      tgt_sr, audio_opt, times, _ = vc.vc_inference(
            sid=0,
            input_audio_path=Path(args.input_path),
            f0_up_key=args.pitch,
            f0_method=args.f0method,
            filter_radius=args.filter_radius,
            index_rate=args.index_rate,
            rms_mix_rate=args.rms_mix_rate,
            protect=args.protect,
      )
      if audio_opt is not None:
            wavfile.write(args.output_file, tgt_sr, audio_opt)
      else:
            print("audio_opt is None")


if __name__ == "__main__":
      load_dotenv("./.env")
      main()
