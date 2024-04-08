from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
import os
import argparse
def add_audio_effects(audio_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping, output_path):

    # Initialize audio effects plugins
    board = Pedalboard(
        [
            HighpassFilter(),
            Compressor(ratio=4, threshold_db=-15),
            Reverb(room_size=reverb_rm_size, dry_level=reverb_dry, wet_level=reverb_wet, damping=reverb_damping)
         ]
    )

    with AudioFile(audio_path) as f:
        with AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:
            # Read one second of audio at a time, until the file is empty:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)

    return output_path

def main():
    parser = argparse.ArgumentParser(description='Reverb', add_help=True)
    parser.add_argument('-i', '--audio_path', type=str, required=True, help='audio file path to add effects to.')
    parser.add_argument('-rsize', '--reverb-size', type=float, default=0.15, help='Reverb room size between 0 and 1')
    parser.add_argument('-rwet', '--reverb-wetness', type=float, default=0.2, help='Reverb wet level between 0 and 1')
    parser.add_argument('-rdry', '--reverb-dryness', type=float, default=0.8, help='Reverb dry level between 0 and 1')
    parser.add_argument('-rdamp', '--reverb-damping', type=float, default=0.7, help='Reverb damping between 0 and 1')
    parser.add_argument('-oformat', '--output-format', type=str, default='flac', help='Output format of audio file. mp3 for smaller file size, wav for best quality')
    args = parser.parse_args()

    # Parse command line arguments
    args = parser.parse_args()

    # Call the function with the provided arguments
    add_audio_effects(
        audio_path=args.audio_path,
        reverb_rm_size=args.reverb_size,
        reverb_wet=args.reverb_wetness,
        reverb_dry=args.reverb_dryness,
        reverb_damping=args.reverb_damping
    )