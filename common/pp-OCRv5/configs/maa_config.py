import inspect
import os
from pathlib import Path

from text_renderer.config import (
    FixedTextColorCfg,
    GeneratorCfg,
    NormPerspectiveTransformCfg,
    RangeTextColorCfg,
    RenderCfg,
)
from text_renderer.corpus import *
from text_renderer.effect import *
from text_renderer.layout.extra_text_line import ExtraTextLineLayout
from text_renderer.layout.same_line import SameLineLayout

# Updated paths relative to MaaData directory
CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
OUT_DIR = CURRENT_DIR.parent / "output"
DATA_DIR = CURRENT_DIR.parent / "output"
BG_DIR = None
CHAR_DIR = CURRENT_DIR.parent / "output" / "zh_CN"
FONT_DIR = CURRENT_DIR.parent / "fonts" / "SubsetOTF" / "CN"
FONT_LIST_DIR = CURRENT_DIR.parent / "fonts"
TEXT_DIR = DATA_DIR / "text"

# Paths to the text files
ZH_CN_DIR = OUT_DIR / "zh_CN"
LONG_TEXT_PATH = ZH_CN_DIR / "long" / "long_wording.txt"
SHORT_TEXT_PATH = ZH_CN_DIR / "short" / "short_wording.txt"
NUMBER_TEXT_PATH = ZH_CN_DIR / "number" / "numbers.txt"

font_cfg = dict(
    font_dir=FONT_DIR,
    font_list_file=FONT_LIST_DIR / "fonts.txt",
    font_size=(14, 23),
)

perspective_transform = NormPerspectiveTransformCfg(25, 25, 3)


def get_long_corpus():
    """Corpus for long text with length 7"""
    return CharCorpus(
        CharCorpusCfg(
            text_paths=[LONG_TEXT_PATH],
            filter_by_chars=True,
            chars_file=CHAR_DIR / "keys.txt",
            length=(7, 8), 
            **font_cfg
        ),
    )


def get_short_corpus():
    """Corpus for short text"""
    return EnumCorpus(
        EnumCorpusCfg(
            text_paths=[SHORT_TEXT_PATH],
            filter_by_chars=True,
            chars_file=CHAR_DIR / "keys.txt",
            num_pick=1,
            **font_cfg
        ),
    )


def get_number_corpus():
    """Corpus for number text"""
    return EnumCorpus(
        EnumCorpusCfg(
            text_paths=[NUMBER_TEXT_PATH],
            filter_by_chars=True,
            chars_file=CHAR_DIR / "keys.txt",
            num_pick=1,
            **font_cfg
        ),
    )


def base_cfg(name, corpus, gray=True, corpus_effects=None, num_images=50):
    return GeneratorCfg(
        num_image=num_images,
        save_dir=OUT_DIR / "render" / name,
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            perspective_transform=perspective_transform,
            gray=gray,
            corpus=corpus,
            corpus_effects=corpus_effects,
        ),
    )


def long_text_config():
    """Configuration for text with length 7-8 characters"""
    # Define color ranges for font colors (blue and brown)
    font_color_ranges = {
        "blue": {
            "fraction": 0.5,
            "l_boundary": [0, 0, 150],
            "h_boundary": [60, 60, 253],
        },
        "brown": {
            "fraction": 0.5,
            "l_boundary": [139, 70, 19],
            "h_boundary": [160, 82, 43],
        },
    }

    # Define line color ranges
    line_color_ranges = {
        "black": {
            "fraction": 0.5,
            "l_boundary": [0, 0, 0],
            "h_boundary": [64, 64, 64],
        },
        "blue": {
            "fraction": 0.5,
            "l_boundary": [0, 0, 150],
            "h_boundary": [60, 60, 253],
        },
    }

    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_long_corpus(),
        gray=False,
        num_images=120000,
        corpus_effects=Effects(
            [
                # Text border with light/dark configuration
                TextBorder(
                    p=0.5,
                    border_width=(1, 2),
                    border_style="solid",
                    blur_radius=0,
                    enable=True,
                    fraction=0.5,
                    light_enable=True,
                    light_fraction=0.5,
                    dark_enable=True,
                    dark_fraction=0.5,
                ),
                
                # Blur effect
                GaussianBlur(
                    p=0.03,
                    blur_limit=(3, 7),
                ),
                
                # Noise effects
                OneOf([
                    Noise(p=0.25, var_limit=(10.0, 50.0)),
                    UniformNoise(p=0.25, intensity_range=(0.1, 0.3)),
                    SaltPepperNoise(p=0.25),
                    PoissonNoise(p=0.25),
                ]),
                
                # Line effects
                OneOf([
                    Line(p=0.1, color_cfg=RangeTextColorCfg(color_ranges=line_color_ranges), thickness=(1, 2)),
                    Line(p=0.1, color_cfg=RangeTextColorCfg(color_ranges=line_color_ranges), thickness=(1, 2)),
                    Line(p=0.1, color_cfg=RangeTextColorCfg(color_ranges=line_color_ranges), thickness=(1, 2)),
                    Line(p=0.1, color_cfg=RangeTextColorCfg(color_ranges=line_color_ranges), thickness=(1, 2)),
                ]),
                
                # Emboss effect
                Emboss(
                    p=0.1,
                    alpha=(0.9, 1.0),
                    strength=(1.5, 1.6),
                ),
                
                # Sharp effect
                BrightnessContrast(
                    p=0.1,
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                ),
                
                # Dropout effects
                OneOf([
                    DropoutRand(),
                    DropoutVertical(thickness=1),
                    DropoutHorizontal(thickness=1),
                ]),
            ]
        ),
    )


def short_text_config():
    """Configuration for short text"""
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_short_corpus(),
        gray=False,
        num_images=60000,  # Generate 60,000 images for short text
        corpus_effects=Effects(
            [
                # Text border
                TextBorder(
                    p=0.3,
                    border_width=(1, 2),
                    border_style="solid",
                    blur_radius=0,
                    enable=True,
                    fraction=0.5,
                    light_enable=True,
                    light_fraction=0.5,
                    dark_enable=True,
                    dark_fraction=0.5,
                ),
                
                # Blur effect
                GaussianBlur(
                    p=0.02,
                    blur_limit=(2, 5),
                ),
                
                # Noise effects
                OneOf([
                    Noise(p=0.2, var_limit=(5.0, 30.0)),
                    UniformNoise(p=0.2, intensity_range=(0.05, 0.2)),
                    SaltPepperNoise(p=0.2),
                    PoissonNoise(p=0.2),
                ]),
                
                # Dropout effects
                OneOf([
                    DropoutRand(),
                    DropoutVertical(thickness=1),
                    DropoutHorizontal(thickness=1),
                ]),
            ]
        ),
    )


def number_text_config():
    """Configuration for number text"""
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_number_corpus(),
        gray=False,
        num_images=20000,  # Generate 20,000 images for number text
        corpus_effects=Effects(
            [
                # Text border
                TextBorder(
                    p=0.4,
                    border_width=(1, 2),
                    border_style="solid",
                    blur_radius=0,
                    enable=True,
                    fraction=0.5,
                    light_enable=True,
                    light_fraction=0.5,
                    dark_enable=True,
                    dark_fraction=0.5,
                ),
                
                # Blur effect
                GaussianBlur(
                    p=0.02,
                    blur_limit=(2, 6),
                ),
                
                # Noise effects
                OneOf([
                    Noise(p=0.15, var_limit=(5.0, 25.0)),
                    UniformNoise(p=0.15, intensity_range=(0.05, 0.15)),
                    SaltPepperNoise(p=0.15),
                    PoissonNoise(p=0.15),
                ]),
                
                # Line effects
                OneOf([
                    Line(p=0.05, color_cfg=FixedTextColorCfg(), thickness=(1, 2)),
                    Line(p=0.05, color_cfg=FixedTextColorCfg(), thickness=(1, 2)),
                ]),
                
                # Dropout effects
                OneOf([
                    DropoutRand(),
                    DropoutVertical(thickness=1),
                    DropoutHorizontal(thickness=1),
                ]),
            ]
        ),
    )


# The configuration file must have a configs variable
configs = [
    long_text_config(),
    short_text_config(),
    number_text_config(),
]
