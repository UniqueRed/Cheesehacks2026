import { useEffect, useMemo, useState } from "react";
import "./WorkspacePages.css";

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

export default function PresentationsPage({
  openUploadFlow,
  onUploadFlowHandled,
  availableProfiles = [],
  presentations = [],
  setPresentations,
  onPresentNow,
}) {
  const [uploadOpen, setUploadOpen] = useState(false);
  const [title, setTitle] = useState("");
  const [file, setFile] = useState(null);
  const [speakerProfile, setSpeakerProfile] = useState("");
  const [errors, setErrors] = useState({});

  const profileOptions = useMemo(
    () => availableProfiles.map((p) => p.name).filter(Boolean),
    [availableProfiles]
  );

  useEffect(() => {
    if (!speakerProfile && profileOptions.length > 0) {
      setSpeakerProfile(profileOptions[0]);
    }
  }, [profileOptions, speakerProfile]);

  useEffect(() => {
    if (openUploadFlow) {
      setUploadOpen(true);
      onUploadFlowHandled?.();
    }
  }, [openUploadFlow, onUploadFlowHandled]);

  const resetForm = () => {
    setTitle("");
    setFile(null);
    setErrors({});
    if (profileOptions.length > 0) setSpeakerProfile(profileOptions[0]);
  };

  const closeModal = () => {
    setUploadOpen(false);
    resetForm();
  };

  const validate = () => {
    const nextErrors = {};
    if (!title.trim()) nextErrors.title = "Title is required";
    if (!file) nextErrors.file = "Presentation file is required";
    if (!speakerProfile) nextErrors.speakerProfile = "Choose a speaker profile";
    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleCreatePresentation = async () => {
    if (!validate()) return;
    try {
      const fileData = await fileToBase64(file);
      const newPresentation = {
        id: crypto.randomUUID(),
        title: title.trim(),
        date: new Date().toLocaleDateString("en-US", {
          month: "short",
          day: "2-digit",
          year: "numeric",
        }),
        duration: "—",
        slides: "—",
        profile: speakerProfile,
        status: "ready",
        fileName: file.name,
        fileData,
        outline: "",
      };
      setPresentations((prev) => [newPresentation, ...prev]);
      closeModal();
    } catch (_) {
      setErrors((prev) => ({ ...prev, file: "Could not process file" }));
    }
  };

  const handleDelete = (id) => {
    setPresentations((prev) => prev.filter((p) => p.id !== id));
  };

  return (
    <div className="workspace-page">
      <div className="page-header">
        <div>
          <h1 className="workspace-title">Presentations</h1>
          <p className="workspace-subtitle">Manage your uploaded presentation library.</p>
        </div>
        <button type="button" className="btn btn-primary" onClick={() => setUploadOpen(true)}>
          Upload Presentation
        </button>
      </div>

      <div className="grid">
        {presentations.length === 0 ? (
          <div className="card">
            <h3 className="card-title">No presentations yet</h3>
            <p className="card-meta">Upload your first presentation to start presenting.</p>
          </div>
        ) : (
          presentations.map((item) => (
            <div className="card" key={item.id}>
              <h3 className="card-title">{item.title}</h3>
              <p className="card-meta">
                {item.date} • {item.profile}
              </p>
              <span className="badge badge-ready">{item.status}</span>
              <div className="card-actions" style={{ marginTop: 10 }}>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => onPresentNow?.(item)}
                >
                  Present Now
                </button>
                <button
                  type="button"
                  className="btn btn-danger"
                  onClick={() => handleDelete(item.id)}
                >
                  Delete
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      {uploadOpen && (
        <div className="overlay" onClick={closeModal}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3 className="card-title">Upload Presentation</h3>

            <div className="field">
              <label>Title</label>
              <input
                className="input"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Presentation title"
              />
              {errors.title ? <small style={{ color: "var(--red)" }}>{errors.title}</small> : null}
            </div>

            <div className="field">
              <label>File (PDF/PPTX)</label>
              <div className="upload-box">
                <input
                  type="file"
                  className="input"
                  accept=".pdf,.pptx,.ppt"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                />
              </div>
              {errors.file ? <small style={{ color: "var(--red)" }}>{errors.file}</small> : null}
            </div>

            <div className="field">
              <label>Speaker profile</label>
              <select
                className="select"
                value={speakerProfile}
                onChange={(e) => setSpeakerProfile(e.target.value)}
              >
                <option value="">Select profile</option>
                {profileOptions.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
              {errors.speakerProfile ? (
                <small style={{ color: "var(--red)" }}>{errors.speakerProfile}</small>
              ) : null}
            </div>

            <div className="card-actions">
              <button type="button" className="btn btn-default" onClick={closeModal}>
                Cancel
              </button>
              <button type="button" className="btn btn-primary" onClick={handleCreatePresentation}>
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
