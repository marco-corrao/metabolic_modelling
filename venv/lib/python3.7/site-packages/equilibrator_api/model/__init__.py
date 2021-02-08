# The MIT License (MIT)
#
# Copyright (c) 2013 Weizmann Institute of Science
# Copyright (c) 2018 Institute for Molecular Systems Biology,
# ETH Zurich
# Copyright (c) 2018 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from typing import Union

try:
    from sbtab.SBtab import SBtabTable, SBtabDocument, SBtabError, read_csv
except ImportError:
    raise ModuleNotFoundError(
        r"In order to use eQuilibrator models, one must install the python "
        r"'sbtab' package first (https://www.sbtab.net/)."
    )


def open_sbtabdoc(filename: Union[str, SBtabDocument]) -> SBtabDocument:
    """Open a file as an SBtabDocument.

    Checks whether it is already an SBtabDocument object, otherwise reads the
    CSV file and returns the parsed object.
    """
    if isinstance(filename, str):
        return read_csv(filename, "pathway")
    elif isinstance(filename, SBtabDocument):
        return filename


from .bounds import Bounds
from .model import StoichiometricModel