import argparse
from pydub import AudioSegment

def combine_audio(audio_paths, output_path, main_gain, inst_gain, output_format):
    main_vocal_audio = AudioSegment.from_file(audio_paths[0], format='flac') + main_gain
    instrumental_audio = AudioSegment.from_file(audio_paths[1], format='flac') + inst_gain
    main_vocal_audio.overlay(instrumental_audio).export(output_path, format=output_format)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine audio files.')
    parser.add_argument('--audio_paths', nargs=2, help='Paths to the audio files to combine.')
    parser.add_argument('--output_path', help='Path to the output file.')
    parser.add_argument('--main_gain', type=int, help='Gain for the main vocal audio.')
    parser.add_argument('--inst_gain', type=int, help='Gain for the instrumental audio.')
    parser.add_argument('--output_format', help='Format of the output file.')
    args = parser.parse_args()

    combine_audio(args.audio_paths, args.output_path, args.main_gain, args.inst_gain, args.output_format)
