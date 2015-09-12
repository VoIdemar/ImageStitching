import cProfile
import pstats

class namedProfile(object):
    """
    Decorator for profiling the specified function.
    """
    def __init__(self, profile_name):
        self.__prof_name = profile_name
        
    def __call__(self, func):
    
        def profile(*args, **kwargs):
            profiler = cProfile.Profile()
            results = profiler.runcall(func, *args, **kwargs)
            
            file_stream = None
            try:
                file_stream = open(self.__prof_name, 'w')
                profiler_stats = pstats.Stats(profiler, stream=file_stream)
                profiler_stats.dump_stats(self.__prof_name + '.raw_stats')
                profiler_stats.sort_stats('cumtime', 'calls')
                profiler_stats.print_stats()
            finally:
                if not file_stream is None:
                    file_stream.close()
            
            return results
    
        return profile