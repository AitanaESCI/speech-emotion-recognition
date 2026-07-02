import React, { useState, useEffect, useRef } from 'react';
import {
  Play, Pause, RotateCcw, CheckCircle, Search,
  HelpCircle, Volume2, ShieldCheck, ChevronRight,
  Settings, RefreshCw, BarChart2, Filter
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function App() {
  const [points, setPoints] = useState([]);
  const [filteredPoints, setFilteredPoints] = useState([]);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [toast, setToast] = useState(null);

  // Filters State
  const [splitFilter, setSplitFilter] = useState('all');
  const [emotionFilter, setEmotionFilter] = useState('all');
  const [annEmotionFilter, setAnnEmotionFilter] = useState('all');
  const [agreementFilter, setAgreementFilter] = useState('all');
  const [auditFilter, setAuditFilter] = useState('all');
  const [searchId, setSearchId] = useState('');

  // View Options State
  const [colorBy, setColorBy] = useState('default'); // 'default', 'iemocap', 'inworld', 'ann_emotion'
  const [hideAudited, setHideAudited] = useState(false);
  const [showBoundaries, setShowBoundaries] = useState(true);
  const [showDiamond, setShowDiamond] = useState(true);
  const [curveCurvature, setCurveCurvature] = useState(1.2);
  const [curveClearance, setCurveClearance] = useState(0.2);

  // Coords & Audit Modes
  const [coordMode, setCoordMode] = useState('old'); // 'old' or 'new'
  const [auditModeEnabled, setAuditModeEnabled] = useState(true);

  // Coordinate resolution helper
  const getPointCoords = (pt) => {
    if (!pt) return { x: 0, y: 0 };
    if (auditModeEnabled) {
      // SQLite override always takes precedence if audited_at is not null
      if (pt.audited_at !== null) {
        return { x: pt.merged_coord_x ?? 0, y: pt.merged_coord_y ?? 0 };
      }
      // Audit Mode overrides
      const ax = pt.audited_new_x;
      const ay = pt.audited_new_y;
      if (ax !== null && ax !== undefined && ay !== null && ay !== undefined) {
        return { x: ax, y: ay };
      }
    }
    // Base coords selection (using original Parquet coords if auditModeEnabled is false)
    if (coordMode === 'old') {
      const bx = pt.orig_coord_x !== undefined && pt.orig_coord_x !== null ? pt.orig_coord_x : pt.merged_coord_x;
      const by = pt.orig_coord_y !== undefined && pt.orig_coord_y !== null ? pt.orig_coord_y : pt.merged_coord_y;
      return { x: bx ?? 0, y: by ?? 0 };
    } else {
      const bx = pt.orig_new_merged_coord_x !== undefined && pt.orig_new_merged_coord_x !== null ? pt.orig_new_merged_coord_x : pt.new_merged_coord_x;
      const by = pt.orig_new_merged_coord_y !== undefined && pt.orig_new_merged_coord_y !== null ? pt.orig_new_merged_coord_y : pt.new_merged_coord_y;
      return { x: bx ?? 0, y: by ?? 0 };
    }
  };

  // Major emotion resolution helper
  const getMajorEmo = (pt) => {
    if (!pt) return '';
    if (auditModeEnabled) {
      if (pt.audited_at !== null && pt.coord_major_emo_override !== null && pt.coord_major_emo_override !== undefined) {
        return pt.coord_major_emo_override;
      }
      if (pt.audited_coord_major_emo_override !== null && pt.audited_coord_major_emo_override !== undefined) {
        return pt.audited_coord_major_emo_override;
      }
    }
    return coordMode === 'old' ? pt.merged_major_emotion : pt.new_merged_major_emotion;
  };

  // Should exclude resolution helper
  const getShouldExclude = (pt) => {
    if (!pt) return false;
    if (auditModeEnabled) {
      if (pt.audited_at !== null) {
        return pt.should_exclude_override || false;
      }
      if (pt.audited_should_exclude_override !== null && pt.audited_should_exclude_override !== undefined) {
        return pt.audited_should_exclude_override === 1 || pt.audited_should_exclude_override === true;
      }
    }
    return false;
  };

  // Dynamic color helper supporting expanded emotion mapping
  const getEmotionColor = (emo) => {
    if (!emo) return '#cbd5e1';
    const e = emo.toLowerCase();
    if (['neutral', 'neu'].includes(e)) return '#64748b'; // Slate/Gray
    if (e === 'calm') return '#14b8a6'; // Teal
    if (['positive', 'happy', 'hap'].includes(e)) return '#10b981'; // Green
    if (e === 'sad') return '#3b82f6'; // Blue
    if (['fearful', 'fear', 'fea'].includes(e)) return '#f97316'; // Orange
    if (['negative', 'angry', 'frustrated', 'ang', 'fru', 'disgusted', 'disgust', 'dis'].includes(e)) return '#f43f5e'; // Red
    if (['surprised', 'surprise', 'sur'].includes(e)) return '#ec4899'; // Pink
    if (e === 'tender') return '#6366f1'; // Indigo
    if (['excited', 'exc'].includes(e)) return '#a855f7'; // Purple
    if (['unclear', 'xxx', 'oth', 'other'].includes(e)) return '#7fb6dbff'; // Slate/Gray-light
    return '#cbd5e1';
  };

  const getEmotionBadgeClass = (emo) => {
    if (!emo) return 'badge-emo-neutral';
    const e = emo.toLowerCase();
    if (['neutral', 'calm'].includes(e)) return 'badge-emo-neutral';
    if (['positive', 'happy', 'tender'].includes(e)) return 'badge-emo-positive';
    if (['sad', 'fearful', 'fear'].includes(e)) return 'badge-emo-sad';
    if (['negative', 'angry', 'frustrated', 'disgusted'].includes(e)) return 'badge-emo-negative';
    return 'badge-emo-neutral';
  };

  // Audio Player State
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const audioRef = useRef(null);

  // Canvas Refs & Coordinates Mapping State
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [canvasSize, setCanvasSize] = useState(600);
  const margin = 50;

  // Dragging State
  const [isDragging, setIsDragging] = useState(false);
  const dragStartCoords = useRef({ x: 0, y: 0 });

  // 1. Fetch Points on Mount
  const fetchPoints = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/api/points`);
      if (!res.ok) throw new Error('Failed to load points from backend');
      const data = await res.json();
      setPoints(data);
      setError(null);
    } catch (err) {
      console.error(err);
      setError('Could not connect to backend server. Make sure the FastAPI backend is running on http://localhost:8000');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPoints();
  }, []);

  // 2. Filter Logic
  useEffect(() => {
    let result = points;

    // Split filter
    if (splitFilter !== 'all') {
      result = result.filter(p => p.split === splitFilter);
    }

    // Emotion filter
    if (emotionFilter !== 'all') {
      result = result.filter(p => p.merged_major_emotion === emotionFilter);
    }

    // Annotated Emotion filter
    if (annEmotionFilter !== 'all') {
      result = result.filter(p => p.ann_emotion === annEmotionFilter);
    }

    // Agreement Ratio filter
    if (agreementFilter !== 'all') {
      result = result.filter(p => {
        const ratio = p.ann_agreement_rate !== undefined && p.ann_agreement_rate !== null
          ? p.ann_agreement_rate
          : (p.ann_agreement_ratio !== undefined ? p.ann_agreement_ratio : (p.ann_agreement / Math.max(p.ann_n_annotators, 1)));
        if (agreementFilter === '100') {
          return ratio === 1.0;
        } else if (agreementFilter === '75-99') {
          return ratio >= 0.75 && ratio < 1.0;
        } else if (agreementFilter === '50-74') {
          return ratio >= 0.50 && ratio < 0.75;
        } else if (agreementFilter === '25-49') {
          return ratio >= 0.25 && ratio < 0.50;
        } else if (agreementFilter === 'lt25') {
          return ratio < 0.25;
        }
        return true;
      });
    }

    // Audit status filter
    if (auditFilter === 'audited') {
      result = result.filter(p => p.audited_at !== null);
    } else if (auditFilter === 'unaudited') {
      result = result.filter(p => p.audited_at === null);
    }

    // Search ID filter (fuzzy string check)
    if (searchId.trim()) {
      result = result.filter(p => p.id.toString().includes(searchId.trim()));
    }

    setFilteredPoints(result);
  }, [points, splitFilter, emotionFilter, annEmotionFilter, agreementFilter, auditFilter, searchId]);

  // Handle Resize of Canvas Container
  useEffect(() => {
    if (!containerRef.current) return;
    const resizeObserver = new ResizeObserver(entries => {
      for (let entry of entries) {
        const size = Math.min(entry.contentRect.width, 700);
        setCanvasSize(size > 0 ? size : 600);
      }
    });
    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, []);

  // Show Toast Message Helper
  const showToast = (message) => {
    setToast(message);
    setTimeout(() => setToast(null), 3000);
  };

  // Convert Math Coordinates [-1.0, 1.0] to Screen Pixels
  const mathToScreen = (x, y, size) => {
    const scale = (size - 2 * margin) / 2;
    const centerX = size / 2;
    const centerY = size / 2;
    return {
      x: centerX + x * scale,
      y: centerY - y * scale // Canvas coordinates go down, math goes up
    };
  };

  // Convert Screen Pixels back to Math Coordinates [-1.0, 1.0]
  const screenToMath = (screenX, screenY, size) => {
    const scale = (size - 2 * margin) / 2;
    const centerX = size / 2;
    const centerY = size / 2;
    return {
      x: (screenX - centerX) / scale,
      y: (centerY - screenY) / scale
    };
  };

  // 3. Canvas Rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear Canvas
    ctx.clearRect(0, 0, canvasSize, canvasSize);

    // Draw grid background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvasSize, canvasSize);

    // Draw grid lines
    ctx.strokeStyle = '#f1f5f9';
    ctx.lineWidth = 1;
    const gridLines = [-0.75, -0.5, -0.25, 0.25, 0.5, 0.75];
    gridLines.forEach(val => {
      // Vertical
      let screen = mathToScreen(val, 0, canvasSize);
      ctx.beginPath();
      ctx.moveTo(screen.x, margin);
      ctx.lineTo(screen.x, canvasSize - margin);
      ctx.stroke();

      // Horizontal
      screen = mathToScreen(0, val, canvasSize);
      ctx.beginPath();
      ctx.moveTo(margin, screen.y);
      ctx.lineTo(canvasSize - margin, screen.y);
      ctx.stroke();
    });

    // Draw Central Axis lines
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1.5;
    const center = mathToScreen(0, 0, canvasSize);

    // Y-Axis
    ctx.beginPath();
    ctx.moveTo(center.x, margin - 10);
    ctx.lineTo(center.x, canvasSize - margin + 10);
    ctx.stroke();

    // X-Axis
    ctx.beginPath();
    ctx.moveTo(margin - 10, center.y);
    ctx.lineTo(canvasSize - margin + 10, center.y);
    ctx.stroke();

    // Axis Tip Labels
    ctx.fillStyle = '#475569';
    ctx.font = '600 12px Outfit';

    // Left [-1, 0] -> Negative
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText('Negative', margin - 8, canvasSize / 2);

    // Right [1, 0] -> Positive
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText('Positive', canvasSize - margin + 8, canvasSize / 2);

    // Top [0, 1] -> Neutral
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('Neutral', canvasSize / 2, margin - 8);

    // Bottom [0, -1] -> Sad / Unclear
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(coordMode === 'new' ? 'Unclear' : 'Sad', canvasSize / 2, canvasSize - margin + 8);

    // Draw Outer Diamond Boundary
    if (showDiamond) {
      ctx.strokeStyle = '#94a3b8';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      const pTop = mathToScreen(0, 1, canvasSize);
      const pRight = mathToScreen(1, 0, canvasSize);
      const pBottom = mathToScreen(0, -1, canvasSize);
      const pLeft = mathToScreen(-1, 0, canvasSize);
      ctx.moveTo(pTop.x, pTop.y);
      ctx.lineTo(pRight.x, pRight.y);
      ctx.lineTo(pBottom.x, pBottom.y);
      ctx.lineTo(pLeft.x, pLeft.y);
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw Separation Curves
    if (showBoundaries) {
      ctx.lineWidth = 2.5;
      ctx.globalAlpha = 0.6; // semi-transparent

      // 1. Top Curve (Neutral) - slate-gray y = a * x^2 + c
      ctx.strokeStyle = '#64748b';
      ctx.beginPath();
      let first = true;
      for (let x = -1.0; x <= 1.0; x += 0.005) {
        const y = curveCurvature * x * x + curveClearance;
        if (Math.abs(x) + Math.abs(y) <= 1.0) {
          const screen = mathToScreen(x, y, canvasSize);
          if (first) {
            ctx.moveTo(screen.x, screen.y);
            first = false;
          } else {
            ctx.lineTo(screen.x, screen.y);
          }
        }
      }
      ctx.stroke();

      // 2. Bottom Curve (Sad) - blue y = -a * x^2 - c
      ctx.strokeStyle = '#3b82f6';
      ctx.beginPath();
      first = true;
      for (let x = -1.0; x <= 1.0; x += 0.005) {
        const y = -curveCurvature * x * x - curveClearance;
        if (Math.abs(x) + Math.abs(y) <= 1.0) {
          const screen = mathToScreen(x, y, canvasSize);
          if (first) {
            ctx.moveTo(screen.x, screen.y);
            first = false;
          } else {
            ctx.lineTo(screen.x, screen.y);
          }
        }
      }
      ctx.stroke();

      // 3. Left Curve (Negative) - red x = -a * y^2 - c
      ctx.strokeStyle = '#f43f5e';
      ctx.beginPath();
      first = true;
      for (let y = -1.0; y <= 1.0; y += 0.005) {
        const x = -curveCurvature * y * y - curveClearance;
        if (Math.abs(x) + Math.abs(y) <= 1.0) {
          const screen = mathToScreen(x, y, canvasSize);
          if (first) {
            ctx.moveTo(screen.x, screen.y);
            first = false;
          } else {
            ctx.lineTo(screen.x, screen.y);
          }
        }
      }
      ctx.stroke();

      // 4. Right Curve (Positive) - green x = a * y^2 + c
      ctx.strokeStyle = '#10b981';
      ctx.beginPath();
      first = true;
      for (let y = -1.0; y <= 1.0; y += 0.005) {
        const x = curveCurvature * y * y + curveClearance;
        if (Math.abs(x) + Math.abs(y) <= 1.0) {
          const screen = mathToScreen(x, y, canvasSize);
          if (first) {
            ctx.moveTo(screen.x, screen.y);
            first = false;
          } else {
            ctx.lineTo(screen.x, screen.y);
          }
        }
      }
      ctx.stroke();

      ctx.globalAlpha = 1.0; // Reset alpha
    }

    // Color definitions for major emotions
    const emoColors = {
      neutral: '#64748b',
      positive: '#10b981',
      sad: '#3b82f6',
      negative: '#f43f5e'
    };

    // Color definitions for annotator emotions
    const annColors = {
      fru: '#f59e0b', // Frustrated
      ang: '#ef4444', // Angry
      sad: '#3b82f6', // Sad
      neu: '#64748b', // Neutral
      exc: '#a855f7', // Excited
      hap: '#22c55e', // Happy
      sur: '#06b6d4', // Surprise
      fea: '#f97316', // Fear
      dis: '#78350f', // Disgust
      xxx: '#94a3b8',
      oth: '#94a3b8'
    };

    // Draw dataset points
    filteredPoints.forEach(p => {
      const isPointAudited = p.audited_at !== null;
      if (isPointAudited && hideAudited) return; // Hide audited toggle

      const coords = getPointCoords(p);
      const screen = mathToScreen(coords.x, coords.y, canvasSize);

      // Determine color based on active color scheme
      let color = '#cbd5e1';
      if (colorBy === 'default') {
        const emo = coordMode === 'old' ? p.merged_major_emotion : p.new_merged_major_emotion;
        color = getEmotionColor(emo);
      } else if (colorBy === 'inverse') {
        const emo = coordMode === 'old' ? p.new_merged_major_emotion : p.merged_major_emotion;
        color = getEmotionColor(emo);
      } else if (colorBy === 'iemocap') {
        color = getEmotionColor(p.reviewed_major_emotion);
      } else if (colorBy === 'inworld') {
        color = getEmotionColor(p.inworld_major_emotion);
      } else {
        color = getEmotionColor(p.ann_emotion);
      }

      // Draw Point circle
      ctx.fillStyle = color;

      // Determine Alpha (Audited points get faint alpha to help focus on unaudited)
      ctx.globalAlpha = isPointAudited ? 0.15 : 0.85;

      ctx.beginPath();
      ctx.arc(screen.x, screen.y, 4, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Reset alpha
    ctx.globalAlpha = 1.0;

    // Draw Selected Point Highlight
    if (selectedPoint) {
      const coords = getPointCoords(selectedPoint);
      const screen = mathToScreen(coords.x, coords.y, canvasSize);

      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(screen.x, screen.y, 8, 0, 2 * Math.PI);
      ctx.stroke();

      // Inner point color
      let selectedColor = '#cbd5e1';
      if (colorBy === 'default') {
        const emo = coordMode === 'old' ? selectedPoint.merged_major_emotion : selectedPoint.new_merged_major_emotion;
        selectedColor = getEmotionColor(emo);
      } else if (colorBy === 'inverse') {
        const emo = coordMode === 'old' ? selectedPoint.new_merged_major_emotion : selectedPoint.merged_major_emotion;
        selectedColor = getEmotionColor(emo);
      } else if (colorBy === 'iemocap') {
        selectedColor = getEmotionColor(selectedPoint.reviewed_major_emotion);
      } else if (colorBy === 'inworld') {
        selectedColor = getEmotionColor(selectedPoint.inworld_major_emotion);
      } else {
        selectedColor = getEmotionColor(selectedPoint.ann_emotion);
      }
      ctx.fillStyle = selectedColor;
      ctx.beginPath();
      ctx.arc(screen.x, screen.y, 5, 0, 2 * Math.PI);
      ctx.fill();
    }

  }, [filteredPoints, selectedPoint, canvasSize, colorBy, hideAudited, showBoundaries, showDiamond, curveCurvature, curveClearance, coordMode, auditModeEnabled]);

  // 4. Interaction Handlers
  const handleMouseMove = (e) => {
    if (isDragging && selectedPoint) {
      handleDrag(e);
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Search for closest point within hover distance threshold (e.g. 10px)
    let closestPoint = null;
    let minDistance = 10;

    filteredPoints.forEach(p => {
      if (p.audited_at !== null && hideAudited) return;

      const coords = getPointCoords(p);
      const screen = mathToScreen(coords.x, coords.y, canvasSize);
      const dist = Math.hypot(screen.x - x, screen.y - y);

      if (dist < minDistance) {
        minDistance = dist;
        closestPoint = p;
      }
    });

    if (closestPoint) {
      setHoveredPoint(closestPoint);
      setTooltipPos({ x: e.clientX - rect.left + 15, y: e.clientY - rect.top - 10 });
    } else {
      setHoveredPoint(null);
    }
  };

  const handleMouseDown = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    let clickedPoint = null;
    let minDistance = 12;

    filteredPoints.forEach(p => {
      if (p.audited_at !== null && hideAudited) return;

      const coords = getPointCoords(p);
      const screen = mathToScreen(coords.x, coords.y, canvasSize);
      const dist = Math.hypot(screen.x - x, screen.y - y);

      if (dist < minDistance) {
        minDistance = dist;
        clickedPoint = p;
      }
    });

    if (clickedPoint) {
      setSelectedPoint(clickedPoint);
      if (auditModeEnabled) {
        setIsDragging(true);
        dragStartCoords.current = getPointCoords(clickedPoint);
      }

      // Auto-play audio when clicking a point
      playAudio(clickedPoint.chunk_compressed_audio_url);
    }
  };

  const handleDrag = (e) => {
    const canvas = canvasRef.current;
    if (!canvas || !selectedPoint) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Convert mouse position to math coordinates
    const mathCoords = screenToMath(x, y, canvasSize);

    // Clamp coordinates between -1.0 and 1.0
    const clampedX = Math.max(-1.0, Math.min(1.0, mathCoords.x));
    const clampedY = Math.max(-1.0, Math.min(1.0, mathCoords.y));

    // Update in selected point local state for drawing feedback
    setSelectedPoint(prev => {
      const updated = { ...prev };

      // Update coordinates
      updated.merged_coord_x = clampedX;
      updated.merged_coord_y = clampedY;
      updated.new_merged_coord_x = clampedX;
      updated.new_merged_coord_y = clampedY;

      if (auditModeEnabled) {
        updated.audited_new_x = clampedX;
        updated.audited_new_y = clampedY;
      }
      return updated;
    });
  };

  const handleMouseUp = async () => {
    if (isDragging && selectedPoint) {
      setIsDragging(false);

      const orig = dragStartCoords.current;
      const currentCoords = getPointCoords(selectedPoint);
      if (orig.x !== currentCoords.x || orig.y !== currentCoords.y) {
        await saveNewCoordinates(selectedPoint.id, currentCoords.x, currentCoords.y);
      }
    }
  };

  // Save Coordinate Update Endpoint
  const saveNewCoordinates = async (id, x, y, override = selectedPoint?.coord_major_emo_override, shouldExcludeOverride = selectedPoint?.should_exclude_override) => {
    try {
      const res = await fetch(`${API_BASE}/api/points/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          merged_coord_x: x,
          merged_coord_y: y,
          coord_major_emo_override: override === "" ? null : override,
          should_exclude_override: shouldExcludeOverride ? true : false
        })
      });

      if (!res.ok) throw new Error('Failed to update coordinates');
      const updated = await res.json();

      // Update our points list in local memory state
      setPoints(prevPoints => prevPoints.map(p => {
        if (p.id === id) {
          return {
            ...p,
            merged_coord_x: x,
            merged_coord_y: y,
            new_merged_coord_x: x,
            new_merged_coord_y: y,
            audited_at: updated.audited_at,
            coord_major_emo_override: updated.coord_major_emo_override,
            should_exclude_override: updated.should_exclude_override
          };
        }
        return p;
      }));

      // Update selected point references
      setSelectedPoint(prev => {
        if (prev && prev.id === id) {
          return {
            ...prev,
            merged_coord_x: x,
            merged_coord_y: y,
            new_merged_coord_x: x,
            new_merged_coord_y: y,
            audited_at: updated.audited_at,
            coord_major_emo_override: updated.coord_major_emo_override,
            should_exclude_override: updated.should_exclude_override
          };
        }
        return prev;
      });

      showToast(`Saved ID ${id}`);
    } catch (err) {
      console.error(err);
      showToast('Error saving updates to database');
    }
  };

  // Save Audit Status Without Changing Coordinates
  const markAsAudited = async () => {
    if (!selectedPoint) return;
    try {
      const res = await fetch(`${API_BASE}/api/points/${selectedPoint.id}/audit`, {
        method: 'POST'
      });
      if (!res.ok) throw new Error('Failed to audit point');
      const updated = await res.json();

      setPoints(prevPoints => prevPoints.map(p => {
        if (p.id === selectedPoint.id) {
          return { ...p, audited_at: updated.audited_at };
        }
        return p;
      }));

      setSelectedPoint(prev => ({ ...prev, audited_at: updated.audited_at }));
      showToast(`Marked ID ${selectedPoint.id} as Audited`);
    } catch (err) {
      console.error(err);
      showToast('Error auditing point');
    }
  };

  // Reset to Original Coordinates Endpoint
  const handleResetCoordinates = async () => {
    if (!selectedPoint) return;
    try {
      const res = await fetch(`${API_BASE}/api/points/${selectedPoint.id}/reset`, {
        method: 'POST'
      });
      if (!res.ok) throw new Error('Failed to reset coordinates');
      const updated = await res.json();

      setPoints(prevPoints => prevPoints.map(p => {
        if (p.id === selectedPoint.id) {
          return {
            ...p,
            merged_coord_x: updated.merged_coord_x,
            merged_coord_y: updated.merged_coord_y,
            audited_at: null
          };
        }
        return p;
      }));

      setSelectedPoint(prev => ({
        ...prev,
        merged_coord_x: updated.merged_coord_x,
        merged_coord_y: updated.merged_coord_y,
        audited_at: null
      }));

      showToast(`Reset coordinates for ID ${selectedPoint.id}`);
    } catch (err) {
      console.error(err);
      showToast('Error resetting coordinates');
    }
  };

  // Skip/Navigate to next Unaudited point
  const handleNextUnaudited = () => {
    const unaudited = filteredPoints.filter(p => p.audited_at === null);
    if (unaudited.length === 0) {
      showToast('No more unaudited points match the filters!');
      return;
    }

    // Find next unaudited point after the current selection
    let nextPoint = unaudited[0];
    if (selectedPoint) {
      const currentIndex = unaudited.findIndex(p => p.id === selectedPoint.id);
      if (currentIndex !== -1 && currentIndex + 1 < unaudited.length) {
        nextPoint = unaudited[currentIndex + 1];
      }
    }

    setSelectedPoint(nextPoint);
    playAudio(nextPoint.chunk_compressed_audio_url);
  };

  // Audio Actions
  const playAudio = (url) => {
    if (!audioRef.current) return;
    audioRef.current.src = url;
    audioRef.current.load();
    audioRef.current.play()
      .then(() => setIsPlaying(true))
      .catch(err => {
        console.error('Audio play failed:', err);
        setIsPlaying(false);
      });
  };

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play()
        .then(() => setIsPlaying(true))
        .catch(err => console.error(err));
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
    }
  };

  const handleProgressBarClick = (e) => {
    if (!audioRef.current || duration === 0) return;
    const bar = e.currentTarget;
    const rect = bar.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const width = rect.width;
    const percentage = clickX / width;
    const newTime = percentage * duration;
    audioRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const formatTime = (time) => {
    if (isNaN(time)) return '0:00';
    const mins = Math.floor(time / 60);
    const secs = Math.floor(time % 60);
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  // Calculate stats
  const totalPoints = points.length;
  const auditedCount = points.filter(p => p.audited_at !== null).count || points.filter(p => p.audited_at !== null).length;
  const auditedPct = totalPoints > 0 ? ((auditedCount / totalPoints) * 100).toFixed(1) : 0;

  // Split-specific stats
  const getSplitStats = (name) => {
    const splitPoints = points.filter(p => p.split === name);
    const splitAudited = splitPoints.filter(p => p.audited_at !== null).length;
    return {
      total: splitPoints.length,
      audited: splitAudited,
      pct: splitPoints.length > 0 ? ((splitAudited / splitPoints.length) * 100).toFixed(0) : 0
    };
  };

  const trainStats = getSplitStats('train');
  const valStats = getSplitStats('validation');
  const testStats = getSplitStats('test');

  return (
    <div className="app-container">
      {/* Invisible Audio Element */}
      <audio
        ref={audioRef}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={() => setIsPlaying(false)}
      />

      {/* Toast Notification */}
      {toast && (
        <div className="status-toast">
          <ShieldCheck size={16} color="#10b981" />
          <span>{toast}</span>
        </div>
      )}

      {/* LEFT SIDEBAR: FILTERS & STATS */}
      <div className="sidebar-left">
        <div>
          <h2 style={{ fontSize: '1.4rem', fontWeight: 700, marginBottom: '0.25rem' }}>EmoTuning</h2>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-light)', marginBottom: '1rem' }}>IEMOCAP Coordinate Labeler</p>
        </div>

        {/* Search */}
        <div className="card" style={{ padding: '1rem' }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">Search Point ID</label>
            <div style={{ position: 'relative' }}>
              <input
                type="text"
                className="input-text"
                placeholder="Enter item index..."
                value={searchId}
                onChange={e => setSearchId(e.target.value)}
                style={{ paddingLeft: '2rem' }}
              />
              <Search size={16} className="text-light" style={{ position: 'absolute', left: '0.75rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-light)' }} />
            </div>
          </div>
        </div>

        {/* Filters Panel */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <div className="card-title">
            <Filter size={16} /> Filters
          </div>

          <div className="form-group">
            <label className="form-label">Dataset Split</label>
            <select className="select-input" value={splitFilter} onChange={e => setSplitFilter(e.target.value)}>
              <option value="all">All Splits</option>
              <option value="train">Train (Ses03, 04, 05)</option>
              <option value="validation">Validation (Ses02)</option>
              <option value="test">Test (Ses01)</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Major Emotion</label>
            <select className="select-input" value={emotionFilter} onChange={e => setEmotionFilter(e.target.value)}>
              <option value="all">All Emotions</option>
              <option value="neutral">Neutral</option>
              <option value="positive">Positive</option>
              <option value="sad">Sad</option>
              <option value="negative">Negative</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Annotated Emotion</label>
            <select className="select-input" value={annEmotionFilter} onChange={e => setAnnEmotionFilter(e.target.value)}>
              <option value="all">All Ann Emotions</option>
              <option value="xxx">xxx (Unknown/Other)</option>
              <option value="fru">fru (Frustrated)</option>
              <option value="neu">neu (Neutral)</option>
              <option value="sad">sad (Sad)</option>
              <option value="ang">ang (Angry)</option>
              <option value="exc">exc (Excited)</option>
              <option value="hap">hap (Happy)</option>
              <option value="sur">sur (Surprise)</option>
              <option value="fea">fea (Fear)</option>
              <option value="oth">oth (Other)</option>
              <option value="dis">dis (Disgust)</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Agreement Ratio</label>
            <select className="select-input" value={agreementFilter} onChange={e => setAgreementFilter(e.target.value)}>
              <option value="all">All Agreements</option>
              <option value="100">100% Agreement (Full)</option>
              <option value="75-99">75% - 99% Agreement</option>
              <option value="50-74">50% - 74% Agreement</option>
              <option value="25-49">25% - 49% Agreement</option>
              <option value="lt25">&lt; 25% Agreement</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Audit Status</label>
            <select className="select-input" value={auditFilter} onChange={e => setAuditFilter(e.target.value)}>
              <option value="all">All Statuses</option>
              <option value="audited">Audited Only</option>
              <option value="unaudited">Unaudited Only</option>
            </select>
          </div>
        </div>

        {/* Stats Panel */}
        <div className="card">
          <div className="card-title">
            <BarChart2 size={16} /> Progress Tracker
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <span style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--border-focus)' }}>{auditedCount}</span>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-light)' }}>of {totalPoints} audited ({auditedPct}%)</span>
            </div>

            <div style={{ width: '100%', height: '8px', backgroundColor: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${auditedPct}%`, backgroundColor: 'var(--border-focus)', borderRadius: '4px' }} />
            </div>

            <div style={{ fontSize: '0.8rem', marginTop: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span className="badge badge-train" style={{ fontSize: '0.65rem' }}>Train</span>
                <span style={{ fontWeight: 550 }}>{trainStats.pct}% ({trainStats.audited}/{trainStats.total})</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span className="badge badge-val" style={{ fontSize: '0.65rem' }}>Val</span>
                <span style={{ fontWeight: 550 }}>{valStats.pct}% ({valStats.audited}/{valStats.total})</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span className="badge badge-test" style={{ fontSize: '0.65rem' }}>Test</span>
                <span style={{ fontWeight: 550 }}>{testStats.pct}% ({testStats.audited}/{testStats.total})</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* CENTER AREA: SCATTER MAP */}
      <div className="main-content">
        <div className="header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h1 style={{ margin: 0 }}>Emotion Map Explorer</h1>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', margin: 0 }}>
              Showing {filteredPoints.length} points. Drag dots to adjust, click to play.
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
            {/* Coordinate System Toggle */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <span className="form-label" style={{ margin: 0, fontSize: '0.7rem' }}>Coords:</span>
              <select
                className="select-input"
                style={{ width: '130px', padding: '0.35rem 0.5rem', fontSize: '0.85rem' }}
                value={coordMode}
                onChange={e => setCoordMode(e.target.value)}
              >
                <option value="old">Old (merged)</option>
                <option value="new">New (new_merged)</option>
              </select>
            </div>

            {/* Audit Mode Checkbox */}
            <label className="checkbox-label" style={{ margin: 0, fontSize: '0.85rem', fontWeight: 550, display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <input
                type="checkbox"
                checked={auditModeEnabled}
                onChange={e => setAuditModeEnabled(e.target.checked)}
              />
              <span>Audit Mode</span>
            </label>

            {/* Color Scheme Dropdown */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <span className="form-label" style={{ margin: 0, fontSize: '0.7rem' }}>Colors:</span>
              <select
                className="select-input"
                style={{ width: '150px', padding: '0.35rem 0.5rem', fontSize: '0.85rem' }}
                value={colorBy}
                onChange={e => setColorBy(e.target.value)}
              >
                <option value="default">
                  {coordMode === 'new' ? 'Default (New)' : 'Default (Old)'}
                </option>
                <option value="inverse">
                  {coordMode === 'new' ? 'Inverse (Old)' : 'Inverse (New)'}
                </option>
                <option value="iemocap">IEMOCAP</option>
                <option value="inworld">Inworld</option>
                <option value="ann_emotion">Annotator</option>
              </select>
            </div>

            <button className="btn btn-secondary" onClick={fetchPoints} title="Refresh dataset" style={{ padding: '0.4rem 0.65rem' }}>
              <RefreshCw size={16} />
            </button>
          </div>
        </div>

        {error && (
          <div className="card" style={{ borderColor: '#ef4444', backgroundColor: '#fee2e2', color: '#991b1b', padding: '1rem' }}>
            {error}
          </div>
        )}

        <div className="canvas-wrapper" ref={containerRef}>
          {loading ? (
            <div className="loading-overlay">
              <div className="spinner" />
              <span>Loading 13,734 audio points...</span>
            </div>
          ) : (
            <div className="canvas-container" style={{ width: canvasSize, height: canvasSize, cursor: auditModeEnabled ? 'crosshair' : 'pointer' }}>
              <canvas
                ref={canvasRef}
                width={canvasSize}
                height={canvasSize}
                className="map-canvas"
                style={{ cursor: auditModeEnabled ? 'crosshair' : 'pointer' }}
                onMouseMove={handleMouseMove}
                onMouseDown={handleMouseDown}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
              />

              {/* Hover Tooltip */}
              {hoveredPoint && (
                <div
                  className="map-tooltip"
                  style={{
                    left: tooltipPos.x,
                    top: tooltipPos.y,
                  }}
                >
                  <div className="tooltip-title">
                    <span>ID {hoveredPoint.id}</span>
                    <span className={`badge badge-${hoveredPoint.split}`}>{hoveredPoint.split}</span>
                  </div>
                  <div className="tooltip-row">
                    <span className="tooltip-label">Speaker:</span>
                    <span className="tooltip-value">{hoveredPoint.speaker_id}</span>
                  </div>
                  <div className="tooltip-row">
                    <span className="tooltip-label">Coord:</span>
                    <span className="tooltip-value">
                      {(() => {
                        const c = getPointCoords(hoveredPoint);
                        return `(${c.x.toFixed(3)}, ${c.y.toFixed(3)})`;
                      })()}
                    </span>
                  </div>
                  <div className="tooltip-row">
                    <span className="tooltip-label">Old Merged Emo:</span>
                    <span className="tooltip-value" style={{ textTransform: 'capitalize' }}>{hoveredPoint.merged_major_emotion}</span>
                  </div>
                  <div className="tooltip-row">
                    <span className="tooltip-label">New Merged Emo:</span>
                    <span className="tooltip-value" style={{ textTransform: 'capitalize' }}>{hoveredPoint.new_merged_major_emotion || 'None'}</span>
                  </div>
                  <div className="tooltip-row">
                    <span className="tooltip-label">IEMOCAP Emo:</span>
                    <span className="tooltip-value" style={{ textTransform: 'capitalize' }}>{hoveredPoint.reviewed_major_emotion || 'None'}</span>
                  </div>
                  <div className="tooltip-row">
                    <span className="tooltip-label">Inworld Emo:</span>
                    <span className="tooltip-value" style={{ textTransform: 'capitalize' }}>{hoveredPoint.inworld_major_emotion || 'None'}</span>
                  </div>
                  <div className="tooltip-row">
                    <span className="tooltip-label">Ann Emo:</span>
                    <span className="tooltip-value" style={{ textTransform: 'uppercase' }}>{hoveredPoint.ann_emotion}</span>
                  </div>
                  <div className="tooltip-row">
                    <span className="tooltip-label">Agreement:</span>
                    <span className="tooltip-value">
                      {hoveredPoint.ann_agreement_rate !== undefined && hoveredPoint.ann_agreement_rate !== null ? (
                        `${(hoveredPoint.ann_agreement_rate * 100).toFixed(0)}%`
                      ) : (
                        `${((hoveredPoint.ann_agreement / Math.max(hoveredPoint.ann_n_annotators, 1)) * 100).toFixed(0)}%`
                      )}
                      {hoveredPoint.ann_agreement !== undefined && hoveredPoint.ann_n_annotators !== undefined && (
                        ` (${hoveredPoint.ann_agreement}/${hoveredPoint.ann_n_annotators})`
                      )}
                    </span>
                  </div>
                  {hoveredPoint.audited_at && (
                    <div style={{ marginTop: '0.25rem', fontSize: '0.7rem', color: '#059669', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
                      <CheckCircle size={10} /> Audited
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* View Toggles Card */}
        <div className="card" style={{ padding: '0.85rem 1.25rem', display: 'flex', flexWrap: 'wrap', gap: '1.5rem', alignItems: 'center' }}>
          <div className="card-title" style={{ borderBottom: 'none', paddingBottom: 0, margin: 0, fontSize: '0.95rem' }}>
            <Settings size={16} /> View Settings
          </div>

          <label className="checkbox-label" style={{ margin: 0 }}>
            <input
              type="checkbox"
              checked={hideAudited}
              onChange={e => setHideAudited(e.target.checked)}
            />
            <span>Hide Audited (otherwise faded out)</span>
          </label>
        </div>

        {/* Separation Boundaries Card */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div className="card-title" style={{ fontSize: '0.95rem', margin: 0, paddingBottom: '0.5rem' }}>
            <Settings size={16} /> Separation Boundaries
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', alignItems: 'center' }}>
            <label className="checkbox-label" style={{ margin: 0 }}>
              <input
                type="checkbox"
                checked={showBoundaries}
                onChange={e => setShowBoundaries(e.target.checked)}
              />
              <span>Show Separation Curves</span>
            </label>

            <label className="checkbox-label" style={{ margin: 0 }}>
              <input
                type="checkbox"
                checked={showDiamond}
                onChange={e => setShowDiamond(e.target.checked)}
              />
              <span>Show Diamond Boundary (|x| + |y| ≤ 1.0)</span>
            </label>
          </div>

          {showBoundaries && (
            <div className="slider-grid" style={{ marginTop: '0.25rem' }}>
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-label">Curvature (a)</span>
                  <span className="slider-value">{curveCurvature.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="3.0"
                  step="0.1"
                  value={curveCurvature}
                  onChange={e => setCurveCurvature(parseFloat(e.target.value))}
                  className="range-input"
                />
              </div>

              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-label">Center Clearance (c)</span>
                  <span className="slider-value">{curveClearance.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="0.8"
                  step="0.05"
                  value={curveClearance}
                  onChange={e => setCurveClearance(parseFloat(e.target.value))}
                  className="range-input"
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* RIGHT SIDEBAR: DETAILS, ADJUSTMENT, AUDIO PLAYER */}
      <div className="sidebar-right">
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, paddingBottom: '0.5rem', borderBottom: '1px solid var(--border-color)' }}>
          Point Inspector
        </h2>

        {selectedPoint ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            {/* ID & Split Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h3 style={{ fontSize: '1.2rem', fontFamily: 'var(--font-title)' }}>Item ID: {selectedPoint.id}</h3>
                <span className="text-light" style={{ fontSize: '0.85rem' }}>Speaker: <b>{selectedPoint.speaker_id}</b></span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', alignItems: 'flex-end' }}>
                <span className={`badge badge-${selectedPoint.split}`}>{selectedPoint.split}</span>
                {selectedPoint.audited_at ? (
                  <span className="badge badge-audited">Audited</span>
                ) : (
                  <span className="badge badge-pending">Pending</span>
                )}
              </div>
            </div>

            {/* Coordinates display & edit */}
            <div className="card" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <span className="form-label" style={{ fontSize: '0.75rem' }}>
                {auditModeEnabled ? 'Coordinate Editor' : 'Coordinates (Read-Only)'}
              </span>
              <div className="drag-coords-display">
                {(() => {
                  const activeCoords = getPointCoords(selectedPoint);
                  return (
                    <>
                      <div className="coord-box">
                        <div className="stat-label">X</div>
                        <div className="coord-val" style={{ color: activeCoords.x >= 0 ? '#10b981' : '#f43f5e' }}>
                          {activeCoords.x.toFixed(4)}
                        </div>
                      </div>
                      <div className="coord-box">
                        <div className="stat-label">Y</div>
                        <div className="coord-val" style={{ color: activeCoords.y >= 0 ? '#3b82f6' : '#ea580c' }}>
                          {activeCoords.y.toFixed(4)}
                        </div>
                      </div>
                    </>
                  );
                })()}
              </div>

              <div style={{ fontSize: '0.75rem', color: 'var(--text-light)', fontStyle: 'italic', textAlign: 'center' }}>
                {auditModeEnabled
                  ? '💡 Tip: Click and drag the dot on the map to modify coordinates.'
                  : '🔒 Turn on Audit Mode to drag and adjust coordinates.'}
              </div>
            </div>

            {/* Audio Player Card */}
            <div className="card" style={{ padding: '1rem' }}>
              <span className="form-label" style={{ marginBottom: '0.5rem', display: 'block' }}>Audio File</span>
              <div className="audio-player-container">
                <div style={{ fontSize: '0.8rem', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-muted)' }}>
                  {selectedPoint.chunk_compressed_audio_url.split('/').pop()}
                </div>
                <div className="audio-controls">
                  <button
                    onClick={togglePlay}
                    className="btn btn-primary"
                    style={{ borderRadius: '50%', width: '36px', height: '36px', padding: 0 }}
                  >
                    {isPlaying ? <Pause size={16} /> : <Play size={16} style={{ marginLeft: '2px' }} />}
                  </button>

                  <div className="audio-progress" onClick={handleProgressBarClick}>
                    <div
                      className="audio-progress-bar"
                      style={{ width: `${duration > 0 ? (currentTime / duration) * 100 : 0}%` }}
                    />
                  </div>

                  <span className="audio-time">
                    {formatTime(currentTime)} / {formatTime(duration)}
                  </span>
                </div>
              </div>
              <div style={{ marginTop: '0.75rem', borderTop: '1px solid var(--border-color)', paddingTop: '0.5rem' }}>
                <label className="checkbox-label" style={{ margin: 0, fontSize: '0.85rem', fontWeight: 500 }}>
                  <input
                    type="checkbox"
                    checked={getShouldExclude(selectedPoint)}
                    disabled={!auditModeEnabled}
                    onChange={async (e) => {
                      const checked = e.target.checked;
                      const activeCoords = getPointCoords(selectedPoint);
                      await saveNewCoordinates(
                        selectedPoint.id,
                        activeCoords.x,
                        activeCoords.y,
                        selectedPoint.coord_major_emo_override,
                        checked
                      );
                    }}
                  />
                  <span>Exclude audio (contains noise/clipping)</span>
                </label>
              </div>
            </div>

            {/* Labels and Agreement card */}
            <div className="card" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <span className="form-label">Emotion Annotations</span>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-light)', fontWeight: 600 }}>MERGED MAJOR</span>
                  <span className={`badge ${getEmotionBadgeClass(selectedPoint.merged_major_emotion)}`} style={{ justifyContent: 'center', marginTop: '0.25rem', textTransform: 'capitalize' }}>
                    {selectedPoint.merged_major_emotion || 'none'}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-light)', fontWeight: 600 }}>ANNOTATOR CONSENSUS</span>
                  <span className={`badge ${getEmotionBadgeClass(selectedPoint.ann_emotion)}`} style={{ justifyContent: 'center', marginTop: '0.25rem', textTransform: 'uppercase' }}>
                    {selectedPoint.ann_emotion || 'none'}
                  </span>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', borderTop: '1px solid var(--border-color)', paddingTop: '0.5rem' }}>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-light)', fontWeight: 600 }}>INWORLD MAJOR</span>
                  <span className={`badge ${getEmotionBadgeClass(selectedPoint.inworld_major_emotion)}`} style={{ justifyContent: 'center', marginTop: '0.25rem', textTransform: 'capitalize' }}>
                    {selectedPoint.inworld_major_emotion || 'none'}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-light)', fontWeight: 600 }}>NEW MERGED MAJOR</span>
                  <span className={`badge ${getEmotionBadgeClass(selectedPoint.new_merged_major_emotion)}`} style={{ justifyContent: 'center', marginTop: '0.25rem', textTransform: 'capitalize' }}>
                    {selectedPoint.new_merged_major_emotion || 'none'}
                  </span>
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid var(--border-color)', paddingTop: '0.5rem', fontSize: '0.85rem' }}>
                <span style={{ color: 'var(--text-light)' }}>Agreement:</span>
                <span style={{ fontWeight: 600 }}>
                  {selectedPoint.ann_agreement_rate !== undefined && selectedPoint.ann_agreement_rate !== null ? (
                    `${(selectedPoint.ann_agreement_rate * 100).toFixed(0)}%`
                  ) : (
                    `${((selectedPoint.ann_agreement / Math.max(selectedPoint.ann_n_annotators, 1)) * 100).toFixed(0)}%`
                  )}
                  {selectedPoint.ann_agreement !== undefined && selectedPoint.ann_n_annotators !== undefined && (
                    ` (${selectedPoint.ann_agreement}/${selectedPoint.ann_n_annotators})`
                  )}
                </span>
              </div>

              <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                <label className="form-label" style={{ fontSize: '0.7rem' }}>Major Emotion Override</label>
                {(() => {
                  const activeCoords = getPointCoords(selectedPoint);
                  const fallbackEmo = (auditModeEnabled && selectedPoint.audited_coord_major_emo_override)
                    ? selectedPoint.audited_coord_major_emo_override
                    : (coordMode === 'old' ? selectedPoint.merged_major_emotion : selectedPoint.new_merged_major_emotion);
                  return (
                    <select
                      className="select-input"
                      value={selectedPoint.coord_major_emo_override || ""}
                      disabled={!auditModeEnabled}
                      onChange={async (e) => {
                        const val = e.target.value === "" ? null : e.target.value;
                        await saveNewCoordinates(
                          selectedPoint.id,
                          activeCoords.x,
                          activeCoords.y,
                          val
                        );
                      }}
                      style={{ padding: '0.4rem 0.5rem', fontSize: '0.85rem' }}
                    >
                      <option value="">None (Use Default: {fallbackEmo})</option>
                      <option value="neutral">Neutral</option>
                      <option value="positive">Positive</option>
                      <option value="sad">Sad</option>
                      <option value="negative">Negative</option>
                    </select>
                  );
                })()}
              </div>
            </div>

            {/* Emotion Probabilities Chart */}
            <div className="card" style={{ padding: '1rem' }}>
              <span className="form-label" style={{ marginBottom: '0.5rem', display: 'block' }}>Probability Densities</span>
              <div className="prob-bars">
                <div className="prob-row">
                  <span style={{ width: '60px', fontWeight: 500 }}>Neutral</span>
                  <div className="prob-bar-container">
                    <div className="prob-bar-fill" style={{ width: `${(selectedPoint.merged_neutral || 0) * 100}%`, backgroundColor: 'var(--color-emo-neutral)' }} />
                  </div>
                  <span style={{ fontFamily: 'monospace', width: '35px', textAlign: 'right' }}>{((selectedPoint.merged_neutral || 0) * 100).toFixed(0)}%</span>
                </div>

                <div className="prob-row">
                  <span style={{ width: '60px', fontWeight: 500 }}>Positive</span>
                  <div className="prob-bar-container">
                    <div className="prob-bar-fill" style={{ width: `${(selectedPoint.merged_positive || 0) * 100}%`, backgroundColor: 'var(--color-emo-positive)' }} />
                  </div>
                  <span style={{ fontFamily: 'monospace', width: '35px', textAlign: 'right' }}>{((selectedPoint.merged_positive || 0) * 100).toFixed(0)}%</span>
                </div>

                <div className="prob-row">
                  <span style={{ width: '60px', fontWeight: 500 }}>Sad</span>
                  <div className="prob-bar-container">
                    <div className="prob-bar-fill" style={{ width: `${(selectedPoint.merged_sad || 0) * 100}%`, backgroundColor: 'var(--color-emo-sad)' }} />
                  </div>
                  <span style={{ fontFamily: 'monospace', width: '35px', textAlign: 'right' }}>{((selectedPoint.merged_sad || 0) * 100).toFixed(0)}%</span>
                </div>

                <div className="prob-row">
                  <span style={{ width: '60px', fontWeight: 500 }}>Negative</span>
                  <div className="prob-bar-container">
                    <div className="prob-bar-fill" style={{ width: `${(selectedPoint.merged_negative || 0) * 100}%`, backgroundColor: 'var(--color-emo-negative)' }} />
                  </div>
                  <span style={{ fontFamily: 'monospace', width: '35px', textAlign: 'right' }}>{((selectedPoint.merged_negative || 0) * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>

            {/* New Merged Emotions Panel */}
            <div className="card" style={{ padding: '1rem' }}>
              <span className="form-label" style={{ marginBottom: '0.5rem', display: 'block' }}>New Merged Emotions</span>
              <div className="prob-bars">
                {[
                  { key: 'new_merged_positive', label: 'Positive', color: 'var(--color-emo-positive)' },
                  { key: 'new_merged_negative', label: 'Negative', color: 'var(--color-emo-negative)' },
                  { key: 'new_merged_neutral', label: 'Neutral', color: 'var(--color-emo-neutral)' },
                  { key: 'new_merged_unclear', label: 'Unclear', color: '#a855f7' }
                ].map(item => {
                  const val = selectedPoint[item.key] || 0;
                  return (
                    <div className="prob-row" key={item.key}>
                      <span style={{ width: '80px', fontWeight: 500 }}>{item.label}</span>
                      <div className="prob-bar-container">
                        <div className="prob-bar-fill" style={{ width: `${val * 100}%`, backgroundColor: item.color }} />
                      </div>
                      <span style={{ fontFamily: 'monospace', width: '35px', textAlign: 'right' }}>{(val * 100).toFixed(0)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Inworld Emotions Panel */}
            <div className="card" style={{ padding: '1rem' }}>
              <span className="form-label" style={{ marginBottom: '0.5rem', display: 'block' }}>Inworld Emotions</span>
              <div className="prob-bars">
                {[
                  { key: 'inworld_emotion_fearful', label: 'Fearful', color: '#f97316' },
                  { key: 'inworld_emotion_neutral', label: 'Neutral', color: '#64748b' },
                  { key: 'inworld_emotion_sad', label: 'Sad', color: '#3b82f6' },
                  { key: 'inworld_emotion_calm', label: 'Calm', color: '#14b8a6' },
                  { key: 'inworld_emotion_angry', label: 'Angry', color: '#f43f5e' },
                  { key: 'inworld_emotion_happy', label: 'Happy', color: '#10b981' },
                  { key: 'inworld_emotion_surprised', label: 'Surprised', color: '#ec4899' },
                  { key: 'inworld_emotion_disgusted', label: 'Disgusted', color: '#78350f' },
                  { key: 'inworld_emotion_tender', label: 'Tender', color: '#6366f1' }
                ].map(item => {
                  const val = selectedPoint[item.key] || 0;
                  return (
                    <div className="prob-row" key={item.key} style={{ marginBottom: '0.25rem' }}>
                      <span style={{ width: '80px', fontWeight: 500 }}>{item.label}</span>
                      <div className="prob-bar-container">
                        <div className="prob-bar-fill" style={{ width: `${val * 100}%`, backgroundColor: item.color }} />
                      </div>
                      <span style={{ fontFamily: 'monospace', width: '35px', textAlign: 'right' }}>{(val * 100).toFixed(0)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Inworld Voice Style Panel */}
            <div className="card" style={{ padding: '1rem' }}>
              <span className="form-label" style={{ marginBottom: '0.5rem', display: 'block' }}>Inworld Voice Style</span>
              <div className="prob-bars">
                {[
                  { key: 'inworld_voice_style_whispering', label: 'Whispering', color: '#cbd5e1' },
                  { key: 'inworld_voice_style_normal', label: 'Normal', color: '#64748b' },
                  { key: 'inworld_voice_style_monotone', label: 'Monotone', color: '#94a3b8' },
                  { key: 'inworld_voice_style_singing', label: 'Singing', color: '#ec4899' },
                  { key: 'inworld_voice_style_mumbling', label: 'Mumbling', color: '#d97706' },
                  { key: 'inworld_voice_style_crying', label: 'Crying', color: '#3b82f6' },
                  { key: 'inworld_voice_style_shouting', label: 'Shouting', color: '#f43f5e' },
                  { key: 'inworld_voice_style_laughing', label: 'Laughing', color: '#10b981' },
                  { key: 'inworld_voice_style_unclear', label: 'Unclear', color: '#a855f7' }
                ].map(item => {
                  const val = selectedPoint[item.key] || 0;
                  return (
                    <div className="prob-row" key={item.key} style={{ marginBottom: '0.25rem' }}>
                      <span style={{ width: '80px', fontWeight: 500 }}>{item.label}</span>
                      <div className="prob-bar-container">
                        <div className="prob-bar-fill" style={{ width: `${val * 100}%`, backgroundColor: item.color }} />
                      </div>
                      <span style={{ fontFamily: 'monospace', width: '35px', textAlign: 'right' }}>{(val * 100).toFixed(0)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Actions panel */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                <button className="btn btn-secondary" onClick={markAsAudited} disabled={!auditModeEnabled || selectedPoint.audited_at !== null}>
                  <CheckCircle size={16} /> Mark Audited
                </button>
                <button className="btn btn-secondary btn-danger" style={{ color: 'white' }} onClick={handleResetCoordinates} disabled={!auditModeEnabled}>
                  <RotateCcw size={16} /> Reset Coords
                </button>
              </div>
              <button
                className="btn btn-primary"
                onClick={handleNextUnaudited}
                style={{ width: '100%', marginTop: '0.25rem' }}
              >
                <span>Next Unaudited Point</span>
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '60%', color: 'var(--text-light)', gap: '0.75rem', textAlign: 'center', padding: '2rem' }}>
            <HelpCircle size={48} strokeWidth={1.5} />
            <p style={{ fontWeight: 550, color: 'var(--text-main)' }}>No Point Selected</p>
            <p style={{ fontSize: '0.85rem' }}>Click any dot on the map to review annotations, listen to the audio chunk, and tune coordinates.</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
