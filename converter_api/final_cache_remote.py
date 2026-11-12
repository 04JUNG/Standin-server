"""Private S3 persistence for verified, registered library FBXs.

Allowlist and revocations live outside the Converter's writable data prefix.
No private pose, job receipt or caller-supplied storage path is registered here.
"""
import os
import re
from urllib.parse import urlsplit


def location(uri):
    p = urlsplit(uri)
    if (p.scheme != 's3' or not re.fullmatch(r'[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]', p.netloc)
            or p.query or p.fragment or not p.path.startswith('/')
            or any(x in ('', '.', '..') for x in p.path[1:].rstrip('/').split('/'))):
        raise ValueError('invalid private final cache S3 location')
    return p.netloc, p.path[1:]


class RemoteStore:
    def __init__(self, data_uri, library_uri, revocations_uri, client=None):
        self.bucket, prefix = location(data_uri)
        self.prefix = prefix.rstrip('/') + '/'
        self.library = location(library_uri)
        self.revocations = location(revocations_uri)
        if any(bucket == self.bucket and key.startswith(self.prefix)
               for bucket, key in (self.library, self.revocations)):
            raise ValueError('control objects must be outside writable data prefix')
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client('s3', config=Config(connect_timeout=2, read_timeout=3,
                retries={'total_max_attempts': 2}))
        self.client = client

    @classmethod
    def from_env(cls):
        data = os.getenv('CONVERTER_FINAL_CACHE_S3_URI')
        if not data:
            return None
        return cls(data, os.getenv('CONVERTER_FINAL_LIBRARY_URI', ''),
                   os.getenv('CONVERTER_FINAL_REVOCATIONS_URI', ''))

    def read_object(self, bucket, key, limit):
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
        except Exception as exc:
            code = getattr(exc, 'response', {}).get('Error', {}).get('Code')
            if str(code) in ('404', 'NoSuchKey', 'NotFound'):
                raise FileNotFoundError('final cache object absent') from None
            raise OSError('private final cache read unavailable') from None
        stream = response['Body']
        try:
            if response.get('ContentLength', limit + 1) > limit:
                raise ValueError('remote final cache object too large')
            data = stream.read(limit + 1)
            if len(data) > limit:
                raise ValueError('remote final cache object too large')
            return data
        except (ValueError, OSError):
            raise
        except Exception:
            raise OSError('private final cache body unavailable') from None
        finally:
            stream.close()

    def read(self, name, limit):
        if re.fullmatch(r'library-[a-f0-9]{64}\.json', name):
            return self.read_object(*self.library, limit)
        if name == 'revoked.json':
            try:
                return self.read_object(*self.revocations, limit)
            except FileNotFoundError:
                raise OSError('required revocation control unavailable') from None
        if not re.fullmatch(r'[a-f0-9]{64}(?:\.[a-f0-9]{64}\.fbx|\.json)', name):
            raise ValueError('invalid remote artifact name')
        return self.read_object(self.bucket, self.prefix + name, limit)

    def publish(self, key, sha, data, metadata):
        if not all(re.fullmatch('[a-f0-9]{64}', x) for x in (key, sha)):
            raise ValueError('invalid remote publication identity')
        # Bytes first, verified receipt last. Immutable blobs cannot mix revisions.
        for name, body in [(key + '.' + sha + '.fbx', data), (key + '.json', metadata)]:
            try:
                self.client.put_object(Bucket=self.bucket, Key=self.prefix + name,
                    Body=body, ServerSideEncryption='AES256',
                    ContentType='application/octet-stream' if name.endswith('.fbx') else 'application/json')
            except Exception:
                raise OSError('private final cache publication unavailable') from None
