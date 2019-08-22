#!/usr/bin/env python3

import array

BYTES = 2   # N bytes arthimetic
MAX = 2 ** (BYTES * 8 - 1) - 1
MIN = - (2 ** (BYTES * 8 - 1)) + 1

CHUNK= 1024

MIN_DELTA = 2

def predictive_adm(samples, deltamax = MAX//21, a = 1):
    """
    Encodes audio bytearray data with Adaptative Delta Modulation. Return a 
    BitStream in a list

    Keyword arguments:
        samples -- Signed 16 bit Audio data in a byte array
        delta - Delta constant of DM modulation. By default it's 1/21 of Max
        Sample value
        a - Sets Integrator decay value. By default it's 1

    """
    raw = array.array('h')
    raw.frombytes(samples)
    stream = []
    integrator = 0
    ndeltah = ndeltal = delta = deltamax // 2
    lastBits = []

    for sample in raw:

        # Adapt Delta value
        if len(lastBits) >= 1:
            if lastBits[-1]:
                ndeltah = min(delta*2, deltamax)
                ndeltal = max(delta//2, MIN_DELTA)
            else:
                ndeltah = max(delta//2, MIN_DELTA)
                ndeltal = min(delta*2, deltamax)

            lastBits.pop()

        highval = integrator + ndeltah
        lowval = integrator - ndeltal

        disthigh = abs(highval - sample)
        distlow = abs(lowval - sample)
        
        # Choose integrator with less diference to sample value
        if disthigh >= distlow:
            stream.append(0)
            integrator = lowval
            delta = ndeltal
        else:
            stream.append(1)
            integrator = highval
            delta = ndeltah
        
        # Clamp to signed 16 bit
        integrator = max(integrator, MIN)
        integrator = min(integrator, MAX)

        integrator = round(integrator * a)
        lastBits.append(stream[-1])

    
    return stream


def decode_adm(stream, deltamax = MAX//21, a = 1):
    """
    Decodes a Adaptative Delta Modulation BitStream in Signed 16 bit Audio data
    in a ByteArray.

    Keywords arguments:
        stream -- List with the BitStream
        delta - Delta constant of DM modulation. By default it's 1/21 of Max
        Sample value
        a - Sets Integrator decay value. By default it's 1
    """

    audio = array.array('h')
    integrator = 0
    ndeltah = ndeltal = delta = deltamax // 2
    lastbits = []
    
    for bit in stream:

        if bit:
            if len(lastbits) >= 1 and lastbits[-1]:
                delta = min(delta*2, deltamax)
            elif len(lastbits) >= 1:
                delta = max(delta//2, MIN_DELTA)

            integrator = integrator + delta
        else:
            if len(lastbits) >= 1 and lastbits[-1]:
                delta = max(delta//2, MIN_DELTA)
            elif len(lastbits) >= 1:
                delta = min(delta*2, deltamax)

            integrator = integrator - delta

        # Clamp to signed 16 bit
        integrator = max(integrator, MIN)
        integrator = min(integrator, MAX)

        integrator = round(integrator * a)

        audio.append(integrator)
        # Store last bits
        lastbits.append(bit)



    return audio.tobytes()

# Main !
if __name__ == '__main__':
    try:
        import pyaudio
    except ImportError:
        print("Wops! We need PyAudio")
        sys.exit(0)

    import random
    import audioop
    import wave
    import sys
    import time
    from math import exp, log10

    random.seed()
    
    p = pyaudio.PyAudio() # Init PyAudio

    wf = wave.open(sys.argv[1], 'rb')
    print("Filename: %s" % sys.argv[1])
    Fm = wf.getframerate()
    print("Sample Rate: %d" % Fm)
    bits = wf.getsampwidth()
    channels = wf.getnchannels()

    samples = wf.readframes(wf.getnframes())    # Get all data from wave file
    wf.close()

    if bits != BYTES:   # Convert to Signed 16 bit data
        samples = audioop.lin2lin(samples, bits, BYTES)
        if bits == 1 and min(samples) >= 0:     # It was 8 bit unsigned !
            samples = audioop.bias(samples, BYTES, MIN)

    if channels > 1:    # Convert to Mono
        samples = audioop.tomono(samples, BYTES, 0.75, 0.25)
    
    # Normalize at 0.9
    maxsample = audioop.max(samples, BYTES)
    samples = audioop.mul(samples, BYTES, MAX * 0.9 / maxsample)

    # Calc A value
    tau = 0.001
    a = exp( -1.0 / (tau * Fm))
    print("A value = %f, Integrator time constant of %fs" % (a, tau))

    # Calc ideal deltamax
    deltamax = round(MAX * (1 - a))
    print("Delta max value = %d" % deltamax)

    # Convert to Delta Modulation
    bitstream = predictive_adm(samples, deltamax, a)

    # Swap random bits (simulate bit errors)
    ERROR_RATE = 0#0.1
    BIT_PROB = 0.5
    tmp = max(BIT_PROB, 1- BIT_PROB)
    print("Error rate %f" % (ERROR_RATE * tmp))

    for i in range(len(bitstream)):
        if random.random() <= ERROR_RATE:
            if random.random() <= BIT_PROB:
                bitstream[i] = 0
            else:
                bitstream[i] = 1


    # Reconvert to PCM
    audio = decode_adm(bitstream, deltamax, a)

    # Play it!
    stream = p.open(format=p.get_format_from_width(BYTES), \
                        channels=1, rate=Fm, output=True)
    data = audio[:CHUNK]
    i = 0
    while i < len(audio):
        stream.write(data)
        i += CHUNK
        data = audio[i:min(i+CHUNK, len(audio))]

    time.sleep(0.5)
    
    stream.stop_stream()
    stream.close()

    p.terminate()

    s = 0.0
    n = 0.0
    for i in range(min(len(samples), len(audio))):
        s = s + samples[i]**2
        n = n +(samples[i] - audio[i])**2

    in_signal = float(s) / len(samples)
    ns_signal = float(n) / len(samples)

    snr = in_signal / ns_signal
    snr_db = 10 * log10(snr)
    print("SNR ratio %f (%f dB) " % (snr, snr_db))
