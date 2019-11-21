#!/usr/bin/env python3
import numpy as np
import skvideo.io
import argparse
from tqdm import tqdm


def image_thresh(a):
    a = a.copy()
    a += np.random.random(a.shape) / 2
    a[a < 0] = 0
    a[a > 255] = 255
    a = np.floor(a)
    return a.astype("uint8")


def get_frame_windows(frames, width=3):
    """Return runs of frames."""
    buf = []
    frames = (frame.astype('float64') for frame in frames)
    while len(buf) < width:
        try:
            buf.append(next(frames))
        except StopIteration:
            break
    for frame in frames:
        buf.append(frame)
        buf = buf[1:]
        yield buf


def frame_weighted_avg(frames, weights):
    frame = next(frames)
    denom = next(weights)
    frame *= denom

    for (f, w) in zip(frames, weights):
        frame += f * w
        denom += w

    return frame / denom


def sinc_interp_frames(windows, factor=2):
    for window in windows:
        window_size = len(window)
        for i in range(factor):
            # fractional offsets, for sinc
            offsets = (np.arange(-window_size // 2,
                                 -window_size // 2 + window_size)
                       - i / factor)
            weights = np.sinc(offsets)

            frame = image_thresh(frame_weighted_avg(iter(window),
                                                    iter(weights)))
            yield frame


def write_frames(frames, filename):
    with skvideo.io.FFmpegWriter(filename) as writer:
        for frame in frames:
            writer.writeFrame(frame)


def main():
    # parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--alpha',
        help='The integral coefficient by which to increase the number of '
        'frames (e.g. alpha=2 yields 2x slow motion).', type=int)
    parser.add_argument('input', help='Video file for input')
    parser.add_argument(
        'output', help='Video file to write the scaled version  to.')

    args = parser.parse_args()

    frame_reader = skvideo.io.vreader(args.input)
    result_frames = sinc_interp_frames(
        get_frame_windows(frame_reader,
                          width=20),
        factor=args.alpha)
    write_frames(tqdm(result_frames), args.output)


if __name__ == '__main__':
    main()
