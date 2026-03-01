import "./WorkspacePages.css";

const RECENT_PRESENTATIONS = [
  { title: "Climate Policy Reform", date: "Feb 26 2026", duration: "12 min" },
  { title: "Q1 Team Retrospective", date: "Feb 20 2026", duration: "8 min" },
];

export default function DashboardHome({ onStartPresenting, onNewPresentation }) {
  return (
    <div className="workspace-page">
      <h1 className="workspace-title">SignBridge</h1>
      <p className="workspace-subtitle">
        Welcome back. Track your speaking flow and launch sessions quickly.
      </p>

      <div className="dashboard-stats">
        <div className="card">
          <div className="stat-label">Presentations</div>
          <div className="stat-value">3</div>
        </div>
        <div className="card">
          <div className="stat-label">Hours Presented</div>
          <div className="stat-value">6.4</div>
        </div>
        <div className="card">
          <div className="stat-label">Calibration Status</div>
          <div className="stat-value">3/5 emotions</div>
        </div>
      </div>

      <div className="quick-actions">
        <button type="button" className="btn btn-primary" onClick={onStartPresenting}>
          Start Presenting
        </button>
        <button type="button" className="btn btn-default" onClick={onNewPresentation}>
          New Presentation
        </button>
      </div>

      <div className="card">
        <h3 className="card-title">Recent Presentations</h3>
        <div className="recent-list">
          {RECENT_PRESENTATIONS.map((item) => (
            <div key={item.title} className="recent-row">
              <div>
                <div className="recent-title">{item.title}</div>
                <div className="recent-meta">{item.date}</div>
              </div>
              <div className="recent-meta">{item.duration}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
