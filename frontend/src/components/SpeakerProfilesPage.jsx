import { useMemo, useState } from "react";
import "./WorkspacePages.css";

export default function SpeakerProfilesPage({ profiles = [], onProfilesChange }) {
  const [modalOpen, setModalOpen] = useState(false);
  const [name, setName] = useState("");
  const [gender, setGender] = useState("Neutral");
  const [accent, setAccent] = useState("Neutral");
  const [tone, setTone] = useState("Confident");
  const [rate, setRate] = useState(60);
  const [editingId, setEditingId] = useState(null);

  const isEditing = useMemo(() => !!editingId, [editingId]);

  const resetForm = () => {
    setName("");
    setGender("Neutral");
    setAccent("Neutral");
    setTone("Confident");
    setRate(60);
    setEditingId(null);
  };

  const openNew = () => {
    resetForm();
    setModalOpen(true);
  };

  const openEdit = (profile) => {
    setEditingId(profile.id);
    setName(profile.name || "");
    setGender(profile.gender || "Neutral");
    setAccent(profile.accent || "Neutral");
    setTone(profile.tone || "Confident");
    setRate(Number(profile.rate || 60));
    setModalOpen(true);
  };

  const saveProfile = () => {
    if (!name.trim()) return;
    if (isEditing) {
      onProfilesChange((prev) =>
        prev.map((p) =>
          p.id === editingId
            ? { ...p, name: name.trim(), gender, accent, tone, rate }
            : p
        )
      );
    } else {
      onProfilesChange((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          name: name.trim(),
          gender,
          accent,
          tone,
          rate,
          isDefault: prev.length === 0,
        },
      ]);
    }
    setModalOpen(false);
    resetForm();
  };

  return (
    <div className="workspace-page">
      <div className="page-header">
        <div>
          <h1 className="workspace-title">Speaker Profiles</h1>
          <p className="workspace-subtitle">Create and manage your speaking profiles.</p>
        </div>
        <button type="button" className="btn btn-primary" onClick={openNew}>
          New Profile
        </button>
      </div>

      <div className="grid">
        {profiles.length === 0 ? (
          <div className="card">
            <h3 className="card-title">No profiles yet</h3>
            <p className="card-meta">Create a speaker profile to personalize presentations.</p>
          </div>
        ) : (
          profiles.map((p) => (
          <div className="card" key={p.id}>
            <div className="page-header" style={{ marginBottom: 8 }}>
              <h3 className="card-title" style={{ margin: 0 }}>{p.name}</h3>
              <div style={{ display: "flex", gap: 6 }}>
                {p.isDefault && <span className="badge badge-ready">Default</span>}
              </div>
            </div>
            <p className="card-meta">
              {p.gender} • {p.accent} • {p.tone}
            </p>
            <div className="recent-meta">Speaking rate: {p.rate}</div>
            <div className="profile-rate">
              <span style={{ width: `${p.rate}%` }} />
            </div>
            <div className="card-actions" style={{ marginTop: 12 }}>
              <button type="button" className="btn btn-default" onClick={() => openEdit(p)}>
                Edit
              </button>
              <button
                type="button"
                className="btn btn-danger"
                onClick={() =>
                  onProfilesChange((prev) => prev.filter((x) => x.id !== p.id))
                }
              >
                Delete
              </button>
            </div>
          </div>
        ))
        )}
      </div>

      {modalOpen && (
        <div
          className="overlay"
          onClick={() => {
            setModalOpen(false);
            resetForm();
          }}
        >
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3 className="card-title">{isEditing ? "Edit Profile" : "New Profile"}</h3>
            <div className="form-grid">
              <div className="field">
                <label>Name</label>
                <input className="input" value={name} onChange={(e) => setName(e.target.value)} />
              </div>
              <div className="field">
                <label>Gender</label>
                <select className="select" value={gender} onChange={(e) => setGender(e.target.value)}>
                  <option>Neutral</option>
                  <option>Feminine</option>
                  <option>Masculine</option>
                </select>
              </div>
              <div className="field">
                <label>Accent</label>
                <select className="select" value={accent} onChange={(e) => setAccent(e.target.value)}>
                  <option>American</option>
                  <option>British</option>
                  <option>Australian</option>
                  <option>Neutral</option>
                </select>
              </div>
              <div className="field">
                <label>Tone</label>
                <select className="select" value={tone} onChange={(e) => setTone(e.target.value)}>
                  <option>Confident</option>
                  <option>Warm</option>
                  <option>Energetic</option>
                  <option>Formal</option>
                  <option>Casual</option>
                </select>
              </div>
            </div>
            <div className="field">
              <label>Speaking rate ({rate})</label>
              <input
                type="range"
                min="20"
                max="100"
                value={rate}
                onChange={(e) => setRate(Number(e.target.value))}
              />
            </div>
            <div className="card-actions">
              <button
                type="button"
                className="btn btn-default"
                onClick={() => {
                  setModalOpen(false);
                  resetForm();
                }}
              >
                Cancel
              </button>
              <button type="button" className="btn btn-primary" onClick={saveProfile}>
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
