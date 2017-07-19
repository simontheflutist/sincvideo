#!/usr/bin/env python3
import numpy as np
import skvideo.io
import argparse
import tempfile
from tqdm import tqdm

def main():
    # parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', help='The integral coefficient by which to increase the number of frames (e.g. alpha=2 yields 2x slow motion).', type=int)
    parser.add_argument('input', help='Video file for input (files longer than ~10s on 20gb memory are not supported yet).')
    parser.add_argument('output', help='Video file to write the scaled version  to.')

    args = parser.parse_args()

    alpha = args.alpha
    frames = skvideo.io.vread(args.input)

    # memorymap the output buffer because this file could be huge
    with tempfile.TemporaryFile() as output_mm:
        output = np.memmap(output_mm, dtype='uint8', mode='w+', shape=(frames.shape[0] * alpha, *frames.shape[1:]))

        for i in tqdm(range(output.shape[0])):
            output[i] = np.zeros(output[i].shape)
            continue
            if i % alpha == 0:
                output[i] = frames[i // alpha]
            else:
                dists = np.abs(np.arange(frames.shape[0]) - i / alpha)
                weights = np.sinc(dists)
                interp = np.tensordot(frames, weights, axes=((0,), (0,)))
                interp[interp < 0] = 0
                interp[interp > 255] = 255
                interp = np.floor(interp)
                output[i] = interp
                output.flush()

        skvideo.io.vwrite(args.output, output)

if __name__ == '__main__':
    main()
