"""Loopback-only review server with byte-range video delivery."""
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import re,shutil
ROOT=Path(__file__).resolve().parents[3]/"validation"
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*a,**k):super().__init__(*a,directory=str(ROOT),**k)
    def send_head(self):
        self.remaining=None
        path=Path(self.translate_path(self.path));header=self.headers.get("Range")
        if not header or not path.is_file():return super().send_head()
        match=re.fullmatch(r"bytes=(\d*)-(\d*)",header);size=path.stat().st_size
        if not match or not any(match.groups()):
            self.send_error(416);return None
        first,last=match.groups()
        start=int(first) if first else max(0,size-int(last))
        end=min(size-1,int(last)) if first and last else size-1
        if start>end or start>=size:
            self.send_response(416);self.send_header("Content-Range",f"bytes */{size}");self.send_header("Content-Length","0");self.end_headers();return None
        f=path.open("rb");f.seek(start);self.remaining=end-start+1
        self.send_response(206);self.send_header("Content-Type",self.guess_type(str(path)))
        self.send_header("Content-Length",str(self.remaining));self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
        self.end_headers();return f
    def end_headers(self):
        self.send_header("Accept-Ranges","bytes");super().end_headers()
    def copyfile(self,source,output):
        if self.remaining is None:return shutil.copyfileobj(source,output)
        left=self.remaining
        while left:
            chunk=source.read(min(left,262144))
            if not chunk:break
            output.write(chunk);left-=len(chunk)
ThreadingHTTPServer(("127.0.0.1",17874),Handler).serve_forever()

