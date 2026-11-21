"""Synthetic bounded-codec tests; no EXE, extracted key/table or user asset fixture."""
import struct
from pathlib import Path
import tempfile
import unittest

from converter.local.contract import LocalError
from converter.local.cs3c import Cipher, Reader, Tables, build_decoder_resource, read_parameters


class CS3CDecoderTests(unittest.TestCase):
    def test_cipher_blocks_are_invertible_without_proprietary_parameters(self):
        cipher = Cipher([(i*2654435761) & 0xffffffff for i in range(1042)], b"authored-test-key")
        for pair in ((0,0), (0x12345678,0xabcdef01), (0xffffffff,0xffffffff)):
            self.assertEqual(cipher.block(*cipher.block(*pair), decrypt=True), pair)

    def test_feedback_and_trailer_decode_authored_plaintext(self):
        cipher = Cipher([(i*2654435761) & 0xffffffff for i in range(1042)], b"authored-test-key")
        plain = b"01234567abcdefgh"
        previous = (42, 73)
        sums = [0,0]
        encoded = bytearray()
        for a,b in struct.iter_unpack(">II", plain):
            pair = (a ^ previous[0], b ^ previous[1])
            sums = [(sums[i]+pair[i]) & 0xffffffff for i in (0,1)]
            encoded.extend(struct.pack(">II", *cipher.block(*pair)))
            previous = (a,b)
        encoded.extend(struct.pack(">II", *cipher.block((sums[0]+42)&0xffffffff, (sums[1]+73)&0xffffffff)))
        self.assertEqual(cipher.decode(encoded), plain)

    def test_truncated_and_oversized_reader_fields_fail_closed(self):
        for action in (lambda: Reader(b"\0").row("<I"),
                       lambda: Reader(struct.pack("<I", 1_000_001)).count(),
                       lambda: Reader(struct.pack("<I", 64)+b"short").string()):
            with self.assertRaises(LocalError):
                action()

    def test_malformed_table_stream_is_not_accepted(self):
        with self.assertRaises(LocalError):
            Tables(b"not-a-table")

    def test_unknown_build_binary_never_becomes_decoder_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp)/"not-modeler.exe", Path(tmp)/"data.json"
            source.write_bytes(b"authored invalid fixture")
            with self.assertRaises(LocalError) as error:
                build_decoder_resource(source, dest)
            self.assertEqual(error.exception.code, "CS3C_DECODER_UNAVAILABLE")
            self.assertFalse(dest.exists())

    def test_invalid_resource_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"resource.json"
            p.write_text('{"schemaVersion":1,"sourceExeSha256":"wrong","parameterPayloadHex":"00"}')
            with self.assertRaises(LocalError) as error:
                read_parameters(p)
            self.assertEqual(error.exception.code, "CS3C_DECODER_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
