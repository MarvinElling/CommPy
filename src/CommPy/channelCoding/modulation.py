import numpy as np


class OOK:
    """
    On-Off Keying (OOK) modulation class.
    This class provides methods to modulate and demodulate binary data using OOK.
    """

    @staticmethod
    def modulate(data, amplitude=1):
        """
        Modulates binary data using OOK.

        Parameters:
        data (list): Binary data to be modulated (0s and 1s).
        amplitude (float): Amplitude of the signal.

        Returns:
        list: Modulated signal.
        """
        return (np.array(data) * amplitude).astype(np.complex128)

    @staticmethod
    def demodulate(signal, threshold=0.5):
        """
        Demodulates an OOK signal back to binary data.

        Parameters:
        signal (list): OOK modulated signal.
        threshold (float): Threshold to determine if the signal is a '1' or '0'.

        Returns:
        list: Demodulated binary data.
        """
        return [1+0j if sample >= threshold else 0 for sample in signal]

