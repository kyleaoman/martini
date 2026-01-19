import os
import numpy as np

"""
The illustris_python toolkit doesn't provide an installable package that we could depend
on, so reproduce the needed utilities here.
"""


def partTypeNum(partType: int | str) -> int:
    """
    Map between common names and numeric particle types.

    Reproduced from the illustris_python toolkit.

    Parameters
    ----------
    partType : int or str
        A descriptor for a particle type. Can be a number (``1`` for dark matter as in
        PartType1) or a string, like ``"dm"`` or ``"darkmatter"``.

    Returns
    -------
    int
        The particle type descriptor converted to a type ID in the range 0-5.
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


def getNumPart(header: dict) -> np.ndarray:
    """
    Calculate number of particles of all types given a snapshot header.

    Reproduced from the illustris_python toolkit.

    Parameters
    ----------
    header : dict
        Dictionary containing header keys and values.

    Returns
    -------
    ~numpy.ndarray
        Array with the number of particles for each of the 6 types.
    """
    nTypes = 6

    nPart = np.zeros(nTypes, dtype=np.int64)
    for j in range(nTypes):
        nPart[j] = header["NumPart_Total"][j] | (
            header["NumPart_Total_HighWord"][j] << 32
        )

    return nPart


def gcPath(basePath: str, snapNum: int, chunkNum: int = 0) -> str:
    """
    Return absolute path to a group catalog HDF5 file (modify as needed).

    Reproduced from the illustris_python toolkit.

    Parameters
    ----------
    basePath : str
        The "base" directory containing the group catalog.

    snapNum : int
        The snapshot number.

    chunkNum : int, optional
        The chunk number for catalogs with multiple files. (Default: ``0``).

    Returns
    -------
    str
        The path to the group catalog file.
    """
    gcPath = basePath + "/groups_%03d/" % snapNum
    filePath1 = gcPath + "groups_%03d.%d.hdf5" % (snapNum, chunkNum)
    filePath2 = gcPath + "fof_subhalo_tab_%03d.%d.hdf5" % (snapNum, chunkNum)

    if os.path.isfile(filePath1):
        return filePath1
    return filePath2


def offsetPath(basePath: str, snapNum: int) -> str:
    """
    Return absolute path to a separate offset file (modify as needed).

    Reproduced from the illustris_python toolkit.

    Paramters
    ---------
    basePath : str
        The base path.

    snapNum : int
        The snapshot number.

    Returns
    -------
    str
        Path to the offset file.
    """
    offsetPath = basePath + "/../postprocessing/offsets/offsets_%03d.hdf5" % snapNum

    return offsetPath


def snapPath(basePath: str, snapNum: int, chunkNum: int = 0) -> str:
    """
    Return absolute path to a snapshot HDF5 file (modify as needed).

    Reproduced from the illustris_python toolkit.

    Paramters
    ---------
    basePath : str
        The base path.

    snapNum : int
        The snapshot number.

    chunkNum : int, optional
        The chunk number for catalogues with multiple files. (Default: ``0``).

    Returns
    -------
    str
        Path to the offset file.
    """
    snapPath = basePath + "/snapdir_" + str(snapNum).zfill(3) + "/"
    filePath = snapPath + "snap_" + str(snapNum).zfill(3)
    filePath += "." + str(chunkNum) + ".hdf5"
    return filePath


def loadSingle(
    basePath: str, snapNum: int, haloID: int = -1, subhaloID: int = -1
) -> dict:
    """
    Return complete group catalog information for one halo or subhalo.

    Reproduced from the illustris_python toolkit.

    Parameters
    ----------
    basePath : str
        The base path.

    snapNum : int
        The snapshot number.

    haloID : int, optional
        The halo ID. (Default: ``-1``).

    subhaloID : int, optional
        The subhalo ID. (Default: ``-1``).

    Returns
    -------
    dict
        The data loaded for the halo/subhalo.
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
    fileNum = int(np.max(np.where(offsets >= 0)))
    groupOffset = offsets[fileNum]

    # load halo/subhalo fields into a dict
    result = {}

    with h5py.File(gcPath(basePath, snapNum, fileNum), "r") as f:
        for haloProp in f[gName].keys():
            result[haloProp] = f[gName][haloProp][groupOffset]

    return result


def loadSubset(
    basePath: str,
    snapNum: int,
    partType: str | int,
    *,
    fields: list[str],
    subset: dict | None = None,
    mdi: list[int | None] | None = None,
    sq: bool = True,
    float32: bool = False,
) -> dict:
    """
    Load a subset of fields for all particles/cells of a given partType.

    Reproduced from the illustris_python toolkit.

    Parameters
    ----------
    basePath : str
        The base path.

    snapNum : int
        The snapshot number.

    partType : str or int
        The particle type.

    fields : list, optional
        The list of data fields to load. (Default: ``None``).

    subset : dict, optional
        If offset and length specified, load only that subset of the partType.
        (Default: ``None``).

    mdi : list, optional
        If mdi is specified, must be a list of integers of the same length as fields,
        giving for each field the multi-dimensional index (on the second dimension) to
        load. For example, fields=['Coordinates', 'Masses'] and mdi=[1, None] returns a
        1D array of y-Coordinates only, together with Masses. (Default: ``None``).

    sq : bool, optional
        If sq is True, return a numpy array instead of a dict if len(fields)==1.
        (Default: ``True``).

    float32 : bool, optional
        If float32 is True, load any float64 datatype arrays directly as float32 (save
        memory). (Default: ``False``).

    Returns
    -------
    dict
        The loaded data.
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

            fileNum = int(np.max(np.where(offsetsThisType >= 0)))
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


def loadHeader(basePath: str, snapNum: int) -> dict:
    """
    Load the group catalog header.

    Reproduced from the illustris_python toolkit.

    Parameters
    ----------
    basePath : str
        The base path.

    snapNum : int
        The snapshot number.

    Returns
    -------
    dict
        The header keys and values.
    """
    import h5py

    with h5py.File(gcPath(basePath, snapNum), "r") as f:
        header = dict(f["Header"].attrs.items())

    return header


def getSnapOffsets(basePath, snapNum, id: int, type: str) -> dict:
    """
    Compute offsets within snapshot for a particular group/subgroup.

    Parameters
    ----------
    basePath : str
        The base path.

    snapNum : int
        The snapshot number

    id : int
        Halo or subhalo ID.

    type : str
        Particle type string.

    Returns
    -------
    dict
        The requested offsets.
    """
    import h5py

    r = {}

    # old or new format
    if "fof_subhalo" in gcPath(basePath, snapNum):
        # use separate 'offsets_nnn.hdf5' files
        with h5py.File(offsetPath(basePath, snapNum), "r") as f:
            groupFileOffsets = f["FileOffsets/" + type][()]
            r["snapOffsets"] = np.transpose(
                f["FileOffsets/SnapByType"][()]
            )  # consistency
    else:
        # load groupcat chunk offsets from header of first file
        with h5py.File(gcPath(basePath, snapNum), "r") as f:
            groupFileOffsets = f["Header"].attrs["FileOffsets_" + type]
            r["snapOffsets"] = f["Header"].attrs["FileOffsets_Snap"]

    # calculate target groups file chunk which contains this id
    groupFileOffsets = int(id) - groupFileOffsets
    fileNum = int(np.max(np.where(groupFileOffsets >= 0)))
    groupOffset = groupFileOffsets[fileNum]

    # load the length (by type) of this group/subgroup from the group catalog
    with h5py.File(gcPath(basePath, snapNum, fileNum), "r") as f:
        r["lenType"] = f[type][type + "LenType"][groupOffset, :]

    # old or new format: load the offset (by type) of this group/subgroup within the
    # snapshot
    if "fof_subhalo" in gcPath(basePath, snapNum):
        with h5py.File(offsetPath(basePath, snapNum), "r") as f:
            r["offsetType"] = f[type + "/SnapByType"][id, :]
    else:
        with h5py.File(gcPath(basePath, snapNum, fileNum), "r") as f:
            r["offsetType"] = f["Offsets"][type + "_SnapByType"][groupOffset, :]

    return r
