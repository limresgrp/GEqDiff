# SDF_handler.py

import warnings
import MDAnalysis as mda
from MDAnalysis.topology.base import TopologyReaderBase
from MDAnalysis.coordinates.base import ReaderBase, WriterBase
from MDAnalysis.lib.util import store_init_arguments, anyopen
from MDAnalysis.core.topology import Topology
from MDAnalysis.core.topologyattrs import (
    Atomnames, Atomtypes, Resids, Resnames, Atomids, Resnums, Segids, ChainIDs
)
import numpy as np

try:
    from rdkit import Chem
    from rdkit import rdBase
    rdBase.DisableLog('rdApp.*')
except ImportError:
    Chem = None

# ---- CLASS 1: The Parser for Topology ----

class SDFParser(TopologyReaderBase):
    """
    Creates a Topology object from a specified molecule in an SDF file.
    This class is responsible for parsing atom and residue information.
    """
    format = ['SDF']

    def parse(self, **kwargs):
        """
        Parses the SDF file and returns a Topology object.
        This is the method MDAnalysis calls to get the topology.
        """
        if Chem is None:
            raise ImportError("RDKit is required for SDF parsing. Please install it: `pip install rdkit`")
            
        mol_index = kwargs.get('mol_index', 0)

        suppl = Chem.SDMolSupplier(self.filename, removeHs=False)
        mol = suppl[mol_index]
        n_atoms = mol.GetNumAtoms()
        atom_names = [atom.GetSymbol() for atom in mol.GetAtoms()]
        
        # Atom level
        attrs = [
            Atomnames(np.array(atom_names)),
            Atomtypes(np.array(atom_names)),
            Atomids(np.arange(n_atoms)),
        ]

        # Residue level
        attrs += [
            Resids(np.ones(1, dtype=np.int32)),
            Resnums(np.ones(1, dtype=np.int32)),
            Resnames(np.array(['MOL'])),
        ]

        # Chain level
        attrs += [
            Segids(np.array(['SYSTEM'])),
            ChainIDs(np.array(['SYSTEM'] * n_atoms)),
        ]
        
        return Topology(n_atoms, attrs=attrs)

# ---- CLASS 2: The Reader for Coordinates ----

class SDFReader(ReaderBase):
    """
    Reads the coordinates from a specified molecule in an SDF file.
    This class is responsible for providing the trajectory data.
    """
    format = 'SDF'
    n_frames = 1

    @store_init_arguments
    def __init__(self, filename, **kwargs):
        if Chem is None:
            raise ImportError("RDKit is required for SDF reading. Please install it: `pip install rdkit`")

        super().__init__(filename, **kwargs)
        mol_index = kwargs.get('mol_index', 0)
        
        # We need n_atoms for the Timestep, which we get from the topology
        # that the Universe has already parsed using SDFParser.
        try:
            self.n_atoms = kwargs['n_atoms']
        except KeyError:
            with SDFParser(self.filename) as p:
                top = p.parse(**kwargs) # Pass kwargs to parser
            self.n_atoms = top.n_atoms
        
        suppl = Chem.SDMolSupplier(self.filename, removeHs=False)
        mol = suppl[mol_index]
        
        self.ts = self._Timestep(self.n_atoms)
        try:
            conformer = mol.GetConformer()
            self.ts.positions = conformer.GetPositions().astype(np.float32)
        except ValueError:
            raise IOError(f"Molecule {mol_index} in {self.filename} has no 3D coordinates.")
    
        self._read_frame(0)

    def Writer(self, filename, **kwargs):
        """Returns a PDBWriter for *filename*.

        Parameters
        ----------
        filename : str
            filename of the output PDB file

        Returns
        -------
        :class:`PDBWriter`

        """
        kwargs.setdefault('multiframe', self.n_frames > 1)
        return SDFWriter(filename, **kwargs)

    def _reopen(self):
        self.close()
        self._pdbfile = anyopen(self.filename, 'rb')
        self.ts.frame = -1

    def _read_frame(self, i):
        if i != 0:
            raise IndexError("Only frame 0 is available for this reader.")
        self.ts.frame = i
        return self.ts
    
    def _read_next_timestep(self):
        if self.ts.frame >= 1:
            return None
        return self._read_frame(0)

    def __len__(self):
        return 1

# ---- CLASS 3: The Writer ----

class SDFWriter(WriterBase):
    format = 'SDF'
    units = {'time': 'ps', 'length': 'Angstrom'}
    
    def __init__(self, filename, **kwargs):
        if Chem is None:
            raise ImportError("RDKit is required for SDF writing. Please install it: `pip install rdkit`")

        self.filename = filename
        # The actual writer object is from RDKit
        self.writer = Chem.SDWriter(self.filename)
        if not self.writer:
            raise IOError(f"Could not open {self.filename} for writing.")
    
    def write(self, obj):
        """Write the current frame of an AtomGroup or Universe to the SDF file."""
        try:
            ag = obj.atoms
        except AttributeError:
            ag = obj
        
        # Create a writable RDKit molecule
        rdmol = Chem.RWMol()

        # Add atoms, keeping a map from MDAnalysis index to RDKit index
        mda_to_rd_idx = {}
        for i, atom in enumerate(ag):
            rd_atom = Chem.Atom(atom.element)
            rd_idx = rdmol.AddAtom(rd_atom)
            mda_to_rd_idx[atom.index] = rd_idx
        
        # Add bonds
        if hasattr(ag.universe, 'bonds') and ag.universe.bonds:
            for bond in ag.universe.bonds:
                a1, a2 = bond.atoms
                if a1.index in mda_to_rd_idx and a2.index in mda_to_rd_idx:
                    rdmol.AddBond(mda_to_rd_idx[a1.index], mda_to_rd_idx[a2.index], Chem.BondType.SINGLE)
        else:
            warnings.warn("No bond information found. Writing a disconnected molecule.")

        # Add coordinates
        conformer = Chem.Conformer(ag.n_atoms)
        for i, pos in enumerate(ag.positions):
            conformer.SetAtomPosition(i, pos)
        rdmol.AddConformer(conformer)
        
        # Use the RDKit writer to write the molecule
        self.writer.write(rdmol.GetMol())

    def close(self):
        """Close the writer."""
        if self.writer:
            self.writer.close()