import ctypes
import json
import os
import platform


def _find_library(lib_name):
    """Find the OpenALPR library."""
    search_paths = []

    # Check environment variable
    if os.environ.get('OPENALPR_LIB_DIR'):
        search_paths.append(os.environ['OPENALPR_LIB_DIR'])

    # Check relative to this file (typical for Windows bundled distribution)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)  # python/
    grandparent_dir = os.path.dirname(parent_dir)  # openalpr_64/
    search_paths.append(grandparent_dir)
    search_paths.append(parent_dir)
    search_paths.append(this_dir)

    if platform.system() == 'Windows':
        lib_filename = lib_name + '.dll'
    elif platform.system() == 'Darwin':
        lib_filename = 'lib' + lib_name + '.dylib'
    else:
        lib_filename = 'lib' + lib_name + '.so'

    for path in search_paths:
        full_path = os.path.join(path, lib_filename)
        if os.path.isfile(full_path):
            return full_path

    # Fallback: let ctypes find it via system PATH
    return lib_name


class Alpr:
    def __init__(self, country, config_file, runtime_dir):
        self._loaded = False

        lib_path = _find_library('openalprpy')
        try:
            self._openalpr = ctypes.cdll.LoadLibrary(lib_path)
        except OSError:
            return

        # This DLL exports camelCase names: initialize, isLoaded, recognizeArray, etc.
        self._openalpr.initialize.restype = ctypes.c_void_p
        self._openalpr.initialize.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]

        self._openalpr.isLoaded.argtypes = [ctypes.c_void_p]
        self._openalpr.isLoaded.restype = ctypes.c_bool

        self._openalpr.recognizeArray.restype = ctypes.c_void_p
        self._openalpr.recognizeArray.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_longlong
        ]

        self._openalpr.recognizeFile.restype = ctypes.c_void_p
        self._openalpr.recognizeFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        self._openalpr.freeJsonMem.argtypes = [ctypes.c_void_p]

        self._openalpr.setCountry.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._openalpr.setPrewarp.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._openalpr.setDefaultRegion.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._openalpr.setDetectRegion.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        self._openalpr.setTopN.argtypes = [ctypes.c_void_p, ctypes.c_int]

        self._openalpr.getVersion.restype = ctypes.c_void_p
        self._openalpr.getVersion.argtypes = [ctypes.c_void_p]

        country_b = country.encode('utf-8') if isinstance(country, str) else country
        config_b = config_file.encode('utf-8') if isinstance(config_file, str) else config_file
        runtime_b = runtime_dir.encode('utf-8') if isinstance(runtime_dir, str) else runtime_dir

        self._instance = self._openalpr.initialize(country_b, config_b, runtime_b)
        self._loaded = self._openalpr.isLoaded(self._instance)

    def is_loaded(self):
        return self._loaded

    def recognize_file(self, file_path):
        """Recognize plates from an image file path."""
        if not self._loaded:
            return {"results": [], "version": 0}

        file_path_b = file_path.encode('utf-8') if isinstance(file_path, str) else file_path
        response_ptr = self._openalpr.recognizeFile(self._instance, file_path_b)
        return self._parse_response(response_ptr)

    def recognize_array(self, byte_array):
        """Recognize plates from encoded image bytes (JPEG, PNG, etc.)."""
        if not self._loaded:
            return {"results": [], "version": 0}

        buf = ctypes.create_string_buffer(bytes(byte_array))
        response_ptr = self._openalpr.recognizeArray(
            self._instance, buf, ctypes.c_longlong(len(byte_array))
        )
        return self._parse_response(response_ptr)

    def _parse_response(self, response_ptr):
        """Parse a JSON response pointer from the DLL."""
        if not response_ptr:
            return {"results": [], "version": 0}
        response_str = ctypes.cast(response_ptr, ctypes.c_char_p).value
        self._openalpr.freeJsonMem(response_ptr)
        if response_str:
            return json.loads(response_str.decode('utf-8'))
        return {"results": [], "version": 0}

    def get_version(self):
        response_ptr = self._openalpr.getVersion(self._instance)
        response_str = ctypes.cast(response_ptr, ctypes.c_char_p).value
        self._openalpr.freeJsonMem(response_ptr)
        return response_str.decode('utf-8') if response_str else "unknown"

    def set_top_n(self, n):
        self._openalpr.setTopN(self._instance, ctypes.c_int(n))

    def set_country(self, country):
        country_b = country.encode('utf-8') if isinstance(country, str) else country
        self._openalpr.setCountry(self._instance, country_b)

    def set_default_region(self, region):
        region_b = region.encode('utf-8') if isinstance(region, str) else region
        self._openalpr.setDefaultRegion(self._instance, region_b)

    def set_detect_region(self, enabled):
        self._openalpr.setDetectRegion(self._instance, ctypes.c_bool(enabled))

    def set_prewarp(self, prewarp):
        prewarp_b = prewarp.encode('utf-8') if isinstance(prewarp, str) else prewarp
        self._openalpr.setPrewarp(self._instance, prewarp_b)

    def unload(self):
        if self._loaded and self._instance:
            self._loaded = False
            self._instance = None

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass
