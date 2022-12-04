import zmq

_PULL_TYPE = b'PULL'
_PUSH_TYPE = b'PUSH'
_UNDEFINED_TYPE = b'UNDEFINED'


class Queue(object):

    def __init__(self, name, gpu_number, protocol="ipc"):
        if protocol == "ipc":
            self._URL = f'ipc://{name}_{gpu_number}.sock'
        else:
            port_number = 5555 + gpu_number
            self._URL = f'tcp://127.0.0.1:{port_number}'

        self._type = _UNDEFINED_TYPE

    def _setup_push(self):
        assert self._type == _UNDEFINED_TYPE
        self._ctx = zmq.Context()
        self._end_point = self._ctx.socket(zmq.PUSH)
        self._end_point.connect(self._URL)
        self._type = _PUSH_TYPE

    def _setup_pull(self):
        assert self._type == _UNDEFINED_TYPE
        self._ctx = zmq.Context()
        self._end_point = self._ctx.socket(zmq.PULL)
        self._end_point.bind(self._URL)
        self._type = _PULL_TYPE

    def _close(self):
        self._end_point.close()
        self._ctx.term()

    def close(self):
        if (self._type == _PUSH_TYPE or self._type == _PULL_TYPE):
            self._close()

    def put(self, msg):
        if (self._type == _UNDEFINED_TYPE):
            self._setup_push()

        assert self._type == _PUSH_TYPE

        self._end_point.send_pyobj(msg, copy=False)

    def get(self):
        if (self._type == _UNDEFINED_TYPE):
            self._setup_pull()
        assert self._type == _PULL_TYPE
        msg = self._end_point.recv_pyobj()
        return msg
