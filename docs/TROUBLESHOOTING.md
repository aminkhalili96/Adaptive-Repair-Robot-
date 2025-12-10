# Troubleshooting Guide

Common issues and solutions for the AARR project.

---

## Installation Issues

### PyBullet fails to install on Mac M1/M2

**Error**:
```
clang: error: the clang compiler does not support...
```

**Solution**:
```bash
# Option 1: Use conda
conda install -c conda-forge pybullet

# Option 2: Run without it
# The app auto-detects and runs in "Mock Mode"
# All UI features work, just no physics simulation
```

---

### `ModuleNotFoundError: No module named 'src'`

**Error**:
```
ModuleNotFoundError: No module named 'src'
```

**Solution**:
```bash
# Run from project root
cd /path/to/robotic_ai

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/robotic_ai"
```

---

### OpenAI API key not found

**Error**:
```
Error: OPENAI_API_KEY not set
```

**Solution**:
```bash
# Option 1: Set environment variable
export OPENAI_API_KEY=sk-your-key-here

# Option 2: Use .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Option 3: Use Ollama (no API key needed)
# Set in config.yaml: agent.provider: ollama
```

---

## Runtime Issues

### Ollama model not responding

**Error**:
```
Connection refused: localhost:11434
```

**Solution**:
```bash
# Start Ollama server
ollama serve

# In another terminal, verify model
ollama list
# Should show: qwen3:14b

# If model missing, pull it
ollama pull qwen3:14b
```

---

### Streamlit port already in use

**Error**:
```
Port 8501 is already in use
```

**Solution**:
```bash
# Kill existing process
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run app/streamlit_app.py --server.port 8502
```

---

### 3D viewer not rendering

**Symptoms**: Blank space where Plotly viewer should be

**Solution**:
1. Refresh the browser (Cmd+R)
2. Clear Streamlit cache: `rm -rf ~/.streamlit/cache`
3. Try different browser (Chrome recommended)

---

### Voice input not working

**Symptoms**: Microphone button doesn't respond

**Causes & Solutions**:
| Cause | Solution |
|-------|----------|
| Browser blocked mic | Click lock icon → Allow microphone |
| No OpenAI key | Voice requires Whisper API — set `OPENAI_API_KEY` |
| HTTPS required | Some browsers require HTTPS for mic access |

---

### Agent returns empty response

**Symptoms**: Chat shows "..." but no response

**Solutions**:
1. Check API key is valid
2. Check Ollama is running (if using local model)
3. Reduce `max_tokens` in config if hitting limits
4. Check `src/agent/supervisor_agent.py` for errors in console

---

## Testing Issues

### Tests fail with import errors

**Error**:
```
ImportError: cannot import name 'X' from 'src.Y'
```

**Solution**:
```bash
# Run tests from project root
cd /path/to/robotic_ai
python -m pytest src/test_*.py -v

# NOT: pytest src/test_*.py (may have path issues)
```

---

### SAM tests fail

**Error**:
```
MobileSAM checkpoint not found
```

**Solution**:
- SAM tests use fallback mode if model not downloaded
- This is expected — fallback still produces valid masks
- To use full SAM: download `mobile_sam.pt` to project root

---

## Performance Issues

### Agent response is slow (>10 sec)

**Causes**:
| Cause | Solution |
|-------|----------|
| Ollama first load | Wait — model loads into memory on first call |
| Large context | Reduce conversation history in agent |
| Slow GPU/CPU | Use smaller model: `qwen3:4b` |

---

### Point cloud generation is slow

**Solution**:
```python
# In config.py, increase voxel size for faster processing
"vision": {
    "pointcloud_voxel_size": 0.01,  # Default 0.005
}
```

Or install `open3d` for GPU acceleration:
```bash
pip install open3d
```

---

## Still Stuck?

1. Check the console/terminal for error messages
2. Review `docs/RUN_TESTS.md` for verification steps
3. Run the test suite: `python -m pytest src/ -v`
4. File an issue on GitHub with:
   - Error message
   - OS version
   - Python version (`python --version`)
   - Steps to reproduce
