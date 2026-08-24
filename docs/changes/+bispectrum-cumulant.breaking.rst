The bispectrum is now computed by Fourier decomposition rather than from the
third-order cumulant, and the previous cumulant-based ``Bispectrum`` API has been
removed. Bispectra are no longer reproduced bit-for-bit from earlier versions,
and code relying on the old attributes or constructor arguments must be updated
to the new ``Bispectrum`` / ``AveragedBispectrum`` interface.
