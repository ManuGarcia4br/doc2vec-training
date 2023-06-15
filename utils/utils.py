import os
import glob


def all_files_iter(directory):
    return glob.iglob(directory + '/**/*', recursive=True)


def extension_validator(wanted_extensions, unwanted_extensions):
    extensions = wanted_extensions or unwanted_extensions
    what_is_valid = (wanted_extensions is not None)

    if extensions is None:
        return lambda x: True

    extensions = set(map(prepend_period_if_necessary, extensions))

    def valid_extension(f):
        _, ext = os.path.splitext(f)
        return ((ext in extensions) == what_is_valid)

    return valid_extension


def filter_filenames(filenames, validator):
    return [
        f for f in filenames
        if not os.path.isdir(f) and validator(f)]


def prepend_period_if_necessary(extension):
    if len(extension) != 0 and extension[0] != '.':
        extension = '.' + extension
    return extension
