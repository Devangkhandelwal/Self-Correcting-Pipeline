/**
 * Inline error panel with a retry action.
 *
 * Added because every failure path in this UI previously went to
 * console.error only, leaving the user with a blank panel and no way to
 * retry short of reloading the page.
 */
export default function ErrorNotice({ title = 'Something went wrong', message, onRetry }) {
  return (
    <div className="error-notice fade-in" role="alert">
      <span className="error-notice__icon" aria-hidden="true">
        ⚠
      </span>
      <div className="error-notice__body">
        <div className="error-notice__title">{title}</div>
        {message && <div className="error-notice__message">{message}</div>}
      </div>
      {onRetry && (
        <button type="button" className="btn btn--ghost" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}
