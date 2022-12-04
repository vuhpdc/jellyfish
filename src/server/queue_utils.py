import fcntl

F_SETPIPE_SZ = 1031
F_GETPIPE_SZ = 1032
PIPE_MAX_SIZE = 1048576


class ParallelQueue(object):
    '''
    ParallelQueue is a wrapper around multiple queues which are scheduled in
    round robin fashion. This is done to overcome the issue of buffer size limit
    per queue. For example, mp.SimpleQueue uses system pipe and it's not straighforward 
    to increase the Pipe's buffer size and also the max size limit. 
    Therefore, using multiple parallel queues might overcome the issue of size limit.
    '''

    def __init__(self, QueueType, count=1):
        self._cur_queue_idx = 0
        self._total_queues = count
        self._queues = [QueueType() for _ in range(count)]

        self._init_queues()

    def _init_queues(self):
        ''' 
        We increase the default pipe  size to max pipe size
        '''
        for queue in self._queues:
            r_fd = queue._reader.fileno()
            w_fd = queue._writer.fileno()
            fcntl.fcntl(r_fd,  F_SETPIPE_SZ, PIPE_MAX_SIZE)
            fcntl.fcntl(w_fd,  F_SETPIPE_SZ, PIPE_MAX_SIZE)

    def close(self):
        [queue.close() for queue in self._queues]

    def put(self, msg):
        idx = self._cur_queue_idx
        self._cur_queue_idx = (self._cur_queue_idx + 1) % self._total_queues
        return self._queues[idx].put(msg)

    def empty(self):
        idx = self._cur_queue_idx
        return self._queues[idx].empty()

    def get(self):
        idx = self._cur_queue_idx
        self._cur_queue_idx = (self._cur_queue_idx + 1) % self._total_queues
        return self._queues[idx].get()
