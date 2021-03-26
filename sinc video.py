#!/usr/bin/env python3
import numpy as np
from scipy.special import sici
import skvideo.io
import argparse
from tqdm import tqdm, trange
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
    # return np.tensordot(
    #     weights.astype('float32'),
    #     np.array(frames).astype('float32'), axes=1
    # )
    out = np.zeros_like(frames[0])
    for i, frame in enumerate(frames):
        out += frame * weights[i]
    return out


def sinc_interp_frames(windows, window_size, ratio, factor=2):
    weights_vectors = create_weights(factor, ratio, window_size)

    for window in windows:
        for i in range(factor):
            weights = weights_vectors[i]
            frame = frame_weighted_avg((window), (weights))
            frame = image_thresh(frame)
            yield frame


def sinc_antiderivative(t):
    si, ci = sici(np.pi * t)
    return si


def create_weights(factor, ratio, window_size):
    """
    Return a list of FACTOR coefficient vectors, each WINDOW_SIZE in length.
    These yield the FACTOR new frames generated for each frame in the input
    to this program.

    RATIO represents shutter speed in the new timeline, assuming
    that the original frames were capture with a 0 degree shutter.
        - If RATIO is 0, then the result is a 0 degree shutter.
        - If RATIO is 0.5, then the result is a 180 degree shutter.
        - If RATIO is greater than 1, than the result is a shutter longer
          than the framerate?
    """
    weights_vectors = []

    for i in range(factor):
        starts = (np.arange(-window_size // 2,
                            -window_size // 2 + window_size)
                  - i / factor)

        # special case for 0 degree shutter
        if ratio == 0:
            weights = np.sinc(starts)
        # otherwise we integrate a little bit
        else:
            half_width = ratio / factor
            weights = (
                sinc_antiderivative(starts + half_width)
                - sinc_antiderivative(starts - half_width)
            )
        weights_vectors.append(weights / np.sum(weights))

    return weights_vectors


def write_frames(frames, filename, limit=240):
    with create_writer(filename) as writer:
        for i, frame in enumerate(frames):
            if limit != -1 and i >= limit:
                break
            writer.writeFrame(frame)


def create_writer(filename):
    fps = 30
    inputdict = {
        '-r': str(fps)
    }
    outputdict = {
        '-r': str(fps),
        # a tutorial had this codec
        '-vcodec': 'libx264',
        # youtube recommends this for 4k uploads,
        '-b': '68M',
        # supposed to reduce deblocking
        '-tune': 'film'
    }

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
        '-r', '--ratio',
        help='Shutter/frame in the new timeline. 0.5 is 180 degrees.',
        required=True,
        type=float)
    parser.add_argument(
        '-n', '--outputframes',
        help='The number of frames to output. If empty, process the whole video',
        default=-1,
        type=int)
    parser.add_argument('input', help='Video file for input')
    parser.add_argument(
        'output', help='Video file to write the scaled version  to.')

    # commandline arguments. also prints error/help and exits here if necessary
    args = parser.parse_args()

    # generates all frames in order
    frame_reader = skvideo.io.FFmpegReader(args.input)

    # buffers the frames into runs of a fixed size
    windows = get_frame_windows(
        frame_reader,
        width=args.buffer
    )

    # at this step each run results in FACTOR frames
    result_frames = sinc_interp_frames(
        windows,
        window_size=args.buffer,
        ratio=args.ratio,
        factor=args.alpha
    )

    # count total number of output frames for reporting progress.
    # it's quite approximate lol
    total_frames = (args.outputframes
                    if args.outputframes != -1
                    else (
                        frame_reader.getShape()[0] - args.buffer // 2
                    ) * args.alpha)

    # iterate through output frames and write them to disk
    write_frames(tqdm(result_frames,
                      total=total_frames),
                 args.output,
                 args.outputframes)


if __name__ == '__main__':
    main()
