
import os, tempfile, shutil, zipfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from vocalchemy_tools import split_stems_demucs, vocal_to_instrument

SOUND_FONT = os.getenv("SOUND_FONT", None)

app = FastAPI(title="VocalAlchemy API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    instrument: str = Form("sax"),
    key: str = Form("C"),
    scale: str = Form("major"),
    bpm: float = Form(100.0),
    harmonize: bool = Form(False),
    choir: bool = Form(False),
):
    tmpdir = Path(tempfile.mkdtemp())
    try:
        inp = tmpdir / file.filename
        with inp.open("wb") as f:
            f.write(await file.read())

        out_path = tmpdir / "converted.wav"
        rendered_path = vocal_to_instrument(
            input_audio=str(inp),
            out_path=str(out_path),
            instrument=instrument,
            key=key,
            scale=scale,
            bpm=bpm,
            harmonize=harmonize,
            choir_mode=choir,
            sf2=SOUND_FONT,
        )
        media = "audio/midi" if str(rendered_path).endswith(".mid") else "audio/wav"
        return FileResponse(rendered_path, media_type=media, filename=Path(rendered_path).name)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

@app.post("/stems")
async def stems(
    file: UploadFile = File(...),
    model: str = Form("htdemucs"),
    export_format: str = Form("wav"),
):
    tmpdir = Path(tempfile.mkdtemp())
    try:
        inp = tmpdir / file.filename
        with inp.open("wb") as f:
            f.write(await file.read())

        out_dir = tmpdir / "stems_out"
        files = split_stems_demucs(
            input_audio=str(inp),
            out_dir=str(out_dir),
            model=model,
            output_format=export_format,
            verbose=False,
        )
        zip_path = tmpdir / "stems.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, arcname=p.name)
        return FileResponse(zip_path, media_type="application/zip", filename="stems.zip")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
