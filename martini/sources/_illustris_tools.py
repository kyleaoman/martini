import os
import numpy as np

"""
The illustris_python toolkit doesn't provide an installable package that we could depend
on, so reproduce the needed utilities here.
"""


def partTypeNum(partType):
    """
    Mapping between common names and numeric particle types.

    Reproduced from the illustris_python toolkit.
    """
    if str(partType).isdigit():
        return int(partType)

    if str(partType).lower() in ["gas", "cells"]:
        return 0
    if str(partType).lower() in ["dm", "darkmatter"]:
        return 1
    if str(partType).lower() in ["tracer", "tracers", "tracermc", "trmc"]:
        return 3
    if str(partType).lower() in ["star", "stars", "stellar"]:
        return 4  # only those with GFM_StellarFormationTime>0
    if str(partType).lower() in ["wind"]:
        return 4  # only those with GFM_StellarFormationTime<0
    if str(partType).lower() in ["bh", "bhs", "blackhole", "blackholes"]:
        return 5

    raise Exception("Unknown particle type name.")


def getNumPart(header):
    """
    Calculate number of particles of all types given a snapshot header.

    Reproduced from the illustris_python toolkit.
    """
    nTypes = 6

    nPart = np.zeros(nTypes, dtype=np.int64)
    for j in range(nTypes):
        nPart[j] = header["NumPart_Total"][j] | (
            header["NumPart_Total_HighWord"][j] << 32
        )

    return nPart


def gcPath(basePath, snapNum, chunkNum=0):
    """
    Return absolute path to a group catalog HDF5 file (modify as needed).

    Reproduced from the illustris_python toolkit.
    """
    gcPath = basePath + "/groups_%03d/" % snapNum
    filePath1 = gcPath + "groups_%03d.%d.hdf5" % (snapNum, chunkNum)
    filePath2 = gcPath + "fof_subhalo_tab_%03d.%d.hdf5" % (snapNum, chunkNum)

    if os.path.isfile(filePath1):
        return filePath1
    return filePath2


def offsetPath(basePath, snapNum):
    """
    Return absolute path to a separate offset file (modify as needed).

    Reproduced from the illustris_python toolkit.
    """
    offsetPath = basePath + "/../postprocessing/offsets/offsets_%03d.hdf5" % snapNum

    return offsetPath


def snapPath(basePath, snapNum, chunkNum=0):
    """Return absolute path to a snapshot HDF5 file (modify as needed)."""
    snapPath = basePath + "/snapdir_" + str(snapNum).zfill(3) + "/"
    filePath = snapPath + "snap_" + str(snapNum).zfill(3)
    filePath += "." + str(chunkNum) + ".hdf5"
    return filePath


def loadSingle(basePath, snapNum, haloID=-1, subhaloID=-1):
    """
    Return complete group catalog information for one halo or subhalo.

    Reproduced from the illustris_python toolkit.
    """
    import h5py

    if (haloID < 0 and subhaloID < 0) or (haloID >= 0 and subhaloID >= 0):
        raise Exception("Must specify either haloID or subhaloID (and not both).")

    gName = "Subhalo" if subhaloID >= 0 else "Group"
    searchID = subhaloID if subhaloID >= 0 else haloID

    # old or new format
    if "fof_subhalo" in gcPath(basePath, snapNum):
        # use separate 'offsets_nnn.hdf5' files
        with h5py.File(offsetPath(basePath, snapNum), "r") as f:
            offsets = f["FileOffsets/" + gName][()]
    else:
        # use header of group catalog
        with h5py.File(gcPath(basePath, snapNum), "r") as f:
            offsets = f["Header"].attrs["FileOffsets_" + gName]

    offsets = searchID - offsets
    fileNum = np.max(np.where(offsets >= 0))
    groupOffset = offsets[fileNum]

    # load halo/subhalo fields into a dict
    result = {}

    with h5py.File(gcPath(basePath, snapNum, fileNum), "r") as f:
        for haloProp in f[gName].keys():
            result[haloProp] = f[gName][haloProp][groupOffset]

    return result


def loadSubset(
    basePath,
    snapNum,
    partType,
    fields=None,
    subset=None,
    mdi=None,
    sq=True,
    float32=False,
):
    """
    Load a subset of fields for all particles/cells of a given partType.
    If offset and length specified, load only that subset of the partType.
    If mdi is specified, must be a list of integers of the same length as fields,
    giving for each field the multi-dimensional index (on the second dimension) to load.
      For example, fields=['Coordinates', 'Masses'] and mdi=[1, None] returns a 1D array
      of y-Coordinates only, together with Masses.
    If sq is True, return a numpy array instead of a dict if len(fields)==1.
    If float32 is True, load any float64 datatype arrays directly as float32
    (save memory).

    Reproduced from the illustris_python toolkit.
    """
    import h5py
    import six

    result = {}

    ptNum = partTypeNum(partType)
    gName = "PartType" + str(ptNum)

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # load header from first chunk
    with h5py.File(snapPath(basePath, snapNum), "r") as f:
        header = dict(f["Header"].attrs.items())
        nPart = getNumPart(header)

        # decide global read size, starting file chunk, and starting file chunk offset
        if subset:
            offsetsThisType = (
                subset["offsetType"][ptNum] - subset["snapOffsets"][ptNum, :]
            )

            fileNum = np.max(np.where(offsetsThisType >= 0))
            fileOff = offsetsThisType[fileNum]
            numToRead = subset["lenType"][ptNum]
        else:
            fileNum = 0
            fileOff = 0
            numToRead = nPart[ptNum]

        result["count"] = numToRead

        if not numToRead:
            # print('warning: no particles of requested type, empty return.')
            return result

        # find a chunk with this particle type
        i = 1
        while gName not in f:
            f = h5py.File(snapPath(basePath, snapNum, i), "r")
            i += 1

        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        for i, field in enumerate(fields):
            # verify existence
            if field not in f[gName].keys():
                raise Exception(
                    "Particle type ["
                    + str(ptNum)
                    + "] does not have field ["
                    + field
                    + "]"
                )

            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = numToRead

            # multi-dimensional index slice load
            if mdi is not None and mdi[i] is not None:
                if len(shape) != 2:
                    raise Exception(
                        "Read error: mdi requested on non-2D field [" + field + "]"
                    )
                shape = [shape[0]]

            # allocate within return dict
            dtype = f[gName][field].dtype
            if dtype == np.float64 and float32:
                dtype = np.float32
            result[field] = np.zeros(shape, dtype=dtype)

    # loop over chunks
    wOffset = 0
    origNumToRead = numToRead

    while numToRead:
        f = h5py.File(snapPath(basePath, snapNum, fileNum), "r")

        # no particles of requested type in this file chunk?
        if gName not in f:
            f.close()
            fileNum += 1
            fileOff = 0
            continue

        # set local read length for this file chunk, truncate to be within the local size
        numTypeLocal = f["Header"].attrs["NumPart_ThisFile"][ptNum]

        numToReadLocal = numToRead

        if fileOff + numToReadLocal > numTypeLocal:
            numToReadLocal = numTypeLocal - fileOff

        # loop over each requested field for this particle type
        for i, field in enumerate(fields):
            # read data local to the current file
            if mdi is None or mdi[i] is None:
                result[field][wOffset : wOffset + numToReadLocal] = f[gName][field][
                    fileOff : fileOff + numToReadLocal
                ]
            else:
                result[field][wOffset : wOffset + numToReadLocal] = f[gName][field][
                    fileOff : fileOff + numToReadLocal, mdi[i]
                ]

        wOffset += numToReadLocal
        numToRead -= numToReadLocal
        fileNum += 1
        fileOff = 0  # start at beginning of all file chunks other than the first

        f.close()

    # verify we read the correct number
    if origNumToRead != wOffset:
        raise Exception(
            "Read ["
            + str(wOffset)
            + "] particles, but was expecting ["
            + str(origNumToRead)
            + "]"
        )

    # only a single field? then return the array instead of a single item dict
    if sq and len(fields) == 1:
        return result[fields[0]]

    return result


def loadHeader(basePath, snapNum):
    """
    Load the group catalog header.

    Reproduced from the illustris_python toolkit.
    """
    import h5py

    with h5py.File(gcPath(basePath, snapNum), "r") as f:
        header = dict(f["Header"].attrs.items())

    return header
