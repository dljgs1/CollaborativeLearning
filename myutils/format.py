"""

    REF: https://www.cnblogs.com/MikeZhang/p/floatNetworkTrans20180320.html

"""

import struct


def float2bytes(f):
    s = struct.pack('d', f)
    return struct.unpack('!Q', s)[0]


def bytes2float(v):
    s = struct.pack('!Q', v)
    return struct.unpack('d', s)[0]


import json

extra_info = {"node_id"}

def unpack_extrainfo(recv):
    try:
        recv = json.loads(recv)
    except:
        pass
    ret = {}
    for extra in extra_info:
        if extra in recv:
            ret[extra] = recv[extra]
    return ret

def unpack_rcvdata(recv):
    try:
        recv = json.loads(recv)
    except:
        pass
    keys = recv["keys"]
    values = recv["values"]
    return {keys[i]: bytes2float(values[i]) for i in range(len(keys))}


def pack_senddata(send):
    keys = list(set(send.keys()) - extra_info)
    values = [float2bytes(send[k]) for k in keys]
    ret = {"keys": keys, "values": values}
    extra = ""
    for e in extra_info:
        if e in send:
            ret[e] = send[e]
            extra += str(send[e])+"、"
    print("pack for:", extra)
    return json.dumps(ret)
