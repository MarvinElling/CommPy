"""commpy package init."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('commpy')
except PackageNotFoundError:  # pragma: no cover -- only when not installed at all
    __version__ = '0.0.0'

# Channel coding: FEC block codes
from ._channelCoding.block.bch import BCHCode
from ._channelCoding.block.crc import CRC, CRCConfig
from ._channelCoding.block.cyclic import CyclicCode
from ._channelCoding.block.hamming import HammingCode
from ._channelCoding.block.reed_solomon import ReedSolomonCode

# Channel coding: convolutional codes + Viterbi decoding
from ._channelCoding.convolutional.encoder import ConvolutionalEncoder
from ._channelCoding.convolutional.trellis import Trellis
from ._channelCoding.convolutional.viterbi import viterbi_decode

# Channel coding: interleaving
from ._channelCoding.interleaving.block import BlockInterleaver
from ._channelCoding.interleaving.convolutional import ConvolutionalInterleaver

# Channel coding: LDPC codes + belief-propagation decoding
from ._channelCoding.ldpc.code import LDPCCode

# Channel coding: polar codes + SC / CRC-aided SCL decoding
from ._channelCoding.polar.code import PolarCode

# Channel coding: turbo codes + iterative BCJR decoding
from ._channelCoding.turbo.turbo import TurboCode

# Channel impairment models (BSC, BEC, AWGN, fading, ...)
from ._channels.channels import Channels

# Galois field arithmetic
from ._fields.binary_extension_field import GF2m
from ._fields.prime_field import PrimeField

# Information theory
from ._informationTheory.formulas import (
    binary_entropy,
    channel_capacity_awgn,
    channel_capacity_bsc,
    channel_capacity_dmc,
    mutual_information,
    shannon_entropy,
)
from ._informationTheory.rate_distortion import rate_distortion_binary
from ._informationTheory.source_coding import (
    arithmetic_decode,
    arithmetic_encode,
    huffman_codes,
    huffman_decode,
    huffman_encode,
)

# MIMO: channel model, Alamouti STBC, spatial-multiplexing detectors, capacity
from ._mimo.capacity import ergodic_mimo_capacity, mimo_capacity
from ._mimo.channel import mimo_awgn, mimo_noise_variance, rayleigh_channel_matrix
from ._mimo.detectors import kbest_detector, ml_detector, mmse_detector, zf_detector
from ._mimo.stbc import alamouti_decode, alamouti_encode

# Digital modulation: generic, Gray-coded engine (preferred for new code)
from ._modulation.base import Modulator

# Digital modulation: equalization
from ._modulation.equalization import mmse_equalizer, zf_equalizer

# Digital modulation: legacy per-scheme classes (kept for backward compatibility)
from ._modulation.legacy import (
    ASK_2_Modulator,
    ASK_4_Modulator,
    BPSK_Modulator,
    OOK_Modulator,
    PSK_8_Modulator,
    QPSK_Modulator,
)
from ._modulation.pam import MPAMModulator
from ._modulation.psk import MPSKModulator

# Digital modulation: pulse shaping
from ._modulation.pulse_shaping import raised_cosine_filter, root_raised_cosine_filter
from ._modulation.qam import MQAMModulator

# Digital modulation: synchronization
from ._modulation.synchronization import (
    costas_loop_bpsk,
    estimate_cfo_mth_power,
    gardner_timing_error,
)

# Networking: queuing theory
from ._networking.queuing import MM1KQueue, MM1Queue, MMcQueue

# OFDM
from ._ofdm.ofdm import OFDMDemodulator, OFDMModulator, papr, papr_ccdf, papr_db

# SDR interoperability: raw IQ and SigMF file I/O
from ._sdr.iq_file import read_iq, read_sigmf, write_iq, write_sigmf

# Link-level simulation: Monte-Carlo BER/FER sweeps
from ._simulation.link_simulation import (
    SimulationResult,
    plot_waterfall,
    simulate_ber,
    simulate_coded_ber,
    simulate_error_rate,
)

# Math utilities
from ._utils.maths import is_prime, modinv

# Waveform synthesis
from ._waves.iq_wave import IQWaveform

__all__ = [
    'CRC',
    'ASK_2_Modulator',
    'ASK_4_Modulator',
    'BCHCode',
    'BPSK_Modulator',
    'BlockInterleaver',
    'CRCConfig',
    'Channels',
    'ConvolutionalEncoder',
    'ConvolutionalInterleaver',
    'CyclicCode',
    'GF2m',
    'HammingCode',
    'IQWaveform',
    'LDPCCode',
    'MM1KQueue',
    'MM1Queue',
    'MMcQueue',
    'MPAMModulator',
    'MPSKModulator',
    'MQAMModulator',
    'Modulator',
    'OFDMDemodulator',
    'OFDMModulator',
    'OOK_Modulator',
    'PSK_8_Modulator',
    'PolarCode',
    'PrimeField',
    'QPSK_Modulator',
    'ReedSolomonCode',
    'SimulationResult',
    'Trellis',
    'TurboCode',
    '__version__',
    'alamouti_decode',
    'alamouti_encode',
    'arithmetic_decode',
    'arithmetic_encode',
    'binary_entropy',
    'channel_capacity_awgn',
    'channel_capacity_bsc',
    'channel_capacity_dmc',
    'costas_loop_bpsk',
    'ergodic_mimo_capacity',
    'estimate_cfo_mth_power',
    'gardner_timing_error',
    'huffman_codes',
    'huffman_decode',
    'huffman_encode',
    'is_prime',
    'kbest_detector',
    'mimo_awgn',
    'mimo_capacity',
    'mimo_noise_variance',
    'ml_detector',
    'mmse_detector',
    'mmse_equalizer',
    'modinv',
    'mutual_information',
    'papr',
    'papr_ccdf',
    'papr_db',
    'plot_waterfall',
    'raised_cosine_filter',
    'rate_distortion_binary',
    'rayleigh_channel_matrix',
    'read_iq',
    'read_sigmf',
    'root_raised_cosine_filter',
    'shannon_entropy',
    'simulate_ber',
    'simulate_coded_ber',
    'simulate_error_rate',
    'viterbi_decode',
    'write_iq',
    'write_sigmf',
    'zf_detector',
    'zf_equalizer',
]
