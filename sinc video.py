#!/usr/bin/env python3
import numpy as np
import skvideo.io
import argparse
from tqdm import tqdm
import PIL
from PIL import Image


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
    frames = (frame.astype('float32') for frame in frames)
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
    return np.tensordot(
        weights.astype('float32'),
        np.array(frames).astype('float32'), axes=1
    ) / sum(weights)


def sinc_interp_frames(windows, factor=2):
    for window in windows:
        window_size = len(window)
        for i in range(factor):
            weights = create_weights_180(i, factor, window_size)
            frame = frame_weighted_avg((window), (weights))
            frame = image_thresh(frame)
            yield frame


def create_weights_old(pos, factor, window_size):
    """
    Pos: position between 0 and 1 that this new frame is sampled from
    """
    offsets = (np.arange(-window_size // 2,
                         -window_size // 2 + window_size)
               - pos / factor) + 0.5
    weights = np.sinc(offsets)
    return weights


def create_weights_180(pos, factor, window_size):
    """
    Pos: position between 0 and 1 that this new frame is sampled from
    """
    n_samples = 20
    weights = np.zeros(window_size)

    for j in range(n_samples):
        offsets = (np.arange(-window_size // 2,
                             -window_size // 2 + window_size)
                   - (pos + 0.5 * j / n_samples) / factor)
        weights += np.sinc(offsets)

    return weights / n_samples


def write_frames(frames, filename, limit=240):
    with create_writer(filename) as writer:
        for i, frame in enumerate(frames):
            if limit != -1 and i >= limit:
                break
            writer.writeFrame(frame)


def create_writer(filename):
    fps = 30
    inputdict = {'-r': str(fps)}
    outputdict = {'-r': str(fps)}
    return skvideo.io.FFmpegWriter(
        filename,
        inputdict=inputdict,
        outputdict=outputdict
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--alpha',
        help='The integral coefficient by which to increase the number of '
        'frames (e.g. alpha=2 yields 2x slow motion).', required=True, type=int)
    parser.add_argument(
        '-b', '--buffer',
        help='The size of the frame buffer used for making new frames. '
        'Bigger is better, but uses more memory.', required=True, type=int)
    parser.add_argument(
        '-n', '--outputframes',
        help='The number of frames to output. If empty, process the whole video',
        default=-1,
        type=int)
    parser.add_argument('input', help='Video file for input')
    parser.add_argument(
        'output', help='Video file to write the scaled version  to.')

    args = parser.parse_args()

    frame_reader = skvideo.io.FFmpegReader(args.input)
    result_frames = sinc_interp_frames(
        get_frame_windows(frame_reader,
                          width=args.buffer),
        factor=args.alpha)

    total_frames = (args.outputframes
                    if args.outputframes != -1
                    else frame_reader.getShape()[0] * args.alpha)
    write_frames(tqdm(result_frames,
                      total=total_frames),
                 args.output,
                 args.outputframes)


if __name__ == '__main__':
    main()
