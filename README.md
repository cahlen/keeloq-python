# keeloq-python
A repository of code that I used to break 160 rounds of the KeeLoq cryptosystem.  Everything is written in python
with the exception of some sage code which is used to converge a system of ANF boolean polynomial equations into
DIMACS CNF form.  Once in this form we can feed the output file to miniSAT which determines if the system is 
satisfiable.  If it is you can pipe the output to a file and parse it with the other python file and check to see
if the correct key was recovered.  

Usually you're going to have to feed miniSAT at least 32 bits of your key as a hint.  If you start taking away bits
of the key then your system is going to be underdefined (espeically if you're only giving it one pair of plaintext/ciphertext).  In this case the system is still satisifiable except you'll get the wrong key back since 
miniSAT stops as soon as it has found the system is satisfiable, and if the system is underdefined it will have 
multiple solutions.

The quick fix to this is to produce another of plaintext/ciphertext under the same key.  Then produce more equations
for the new plaintext/ciphertext, and all the intermediate variables.  The only thing that should remain the same is
the actual key variables.  Of course this system takes longer to solve, but you get the correct key back each time.  I experimented with this and was able to cut the key bit hints down to 25 bits with two pairs of plaintext/ciphertext -- however it took 14 hours to solve this system.

## Phase 1 (2026 modernization) — Quick start

Install (Linux, RTX 5090 optional, Docker recommended for legacy compat tests):

    uv sync --all-extras

Run the full test suite:

    uv run pytest -m "not slow"

Run a 64-round attack with four plaintext/ciphertext pairs and no hints:

    # First, compute the two ciphertexts under the known key:
    KEY=0011010011011111100101100001110000011101100111001000001101110100
    CT1=$(uv run keeloq encrypt --rounds 64 --plaintext 01100010100101110000101011100011 --key $KEY)
    CT2=$(uv run keeloq encrypt --rounds 64 --plaintext 11010011100101010000111100001010 --key $KEY)
    # Then run the attack:
    uv run keeloq attack \
        --rounds 64 \
        --pair "01100010100101110000101011100011:$CT1" \
        --pair "11010011100101010000111100001010:$CT2" \
        --encoder xor --solver cryptominisat --timeout 120

Reproduce the 2015 README result (160 rounds, 25 hints, 2 pairs):

    uv run keeloq benchmark

See `docs/superpowers/specs/2026-04-22-phase1-foundation-design.md` for the
design and `docs/superpowers/plans/2026-04-22-phase1-foundation-plan.md` for
the implementation plan.
