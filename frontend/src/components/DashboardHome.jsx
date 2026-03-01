import "./WorkspacePages.css";

const CALIBRATED_SET_KEY = "facialEmotionCalibratedSet";
const TOTAL_EMOTIONS = 5;

function getCalibratedEmotionsCount() {
  try {
    const raw = localStorage.getItem(CALIBRATED_SET_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    const unique = new Set(Array.isArray(parsed) ? parsed : []);
    return unique.size;
  } catch (_) {
    return 0;
  }
}

export default function DashboardHome({
  onStartPresenting,
  onNewPresentation,
  presentations = [],
  speakerProfiles = [],
}) {
  const calibratedCount = getCalibratedEmotionsCount();
  const recentPresentations = presentations.slice(0, 3);

  return (
    <div className="workspace-page">
      <h1 className="workspace-title">
        Sign<span style={{ color: "var(--accent)" }}>Speak</span>
      </h1>
      <p className="workspace-subtitle">
        Welcome back. Track your speaking flow and launch sessions quickly.
      </p>
      <div className="dashboard-stats">
        <div className="card">
          <div className="stat-label">Presentations</div>
          <div className="stat-value">{presentations.length}</div>
        </div>
        <div className="card">
          <div className="stat-label">Calibration Status</div>

          <div className="stat-value">
            {calibratedCount}/{TOTAL_EMOTIONS} emotions
          </div>
        </div>
      </div>
      <div className="quick-actions">
        <button
          type="button"
          className="btn btn-primary"
          onClick={onStartPresenting}
        >
          Start Presenting
        </button>
        <button
          type="button"
          className="btn btn-default"
          onClick={onNewPresentation}
        >
          New Presentation
        </button>
      </div>

      <div className="card">
        <h3 className="card-title">Recent Presentations</h3>

        <div className="recent-list">
          {recentPresentations.length === 0 ? (
            <div className="recent-row">
              <div>
                <div className="recent-title">No presentations yet</div>

                <div className="recent-meta">
                  Upload your first presentation to see activity here.
                </div>
              </div>
            </div>
          ) : (
            recentPresentations.map((item) => (
              <div key={item.id} className="recent-row">
                <div>
                  <div className="recent-title">{item.title}</div>

                  <div className="recent-meta">{item.date || "—"}</div>
                </div>

                <div className="recent-meta">{item.duration || "—"}</div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
