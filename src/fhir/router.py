from fastapi import APIRouter, Depends, Response, HTTPException
from sqlalchemy.orm import Session

from src.database import get_db
from .service import build_fhir_bundle_for_user

router = APIRouter(
    prefix="/api/fhir",
)

@router.get("/export")
def export_fhir_for_user(
    user_email: str,
    db: Session = Depends(get_db),
):
    # xml로 export
    xml_bytes = build_fhir_bundle_for_user(user_email=user_email, db=db)
    if not xml_bytes:
        raise HTTPException(
            status_code=404,
            detail="no data to export"
        )

    return Response(
        content=xml_bytes,
        headers={
            "Content-Disposition": f'attachment; filename="health_{user_email}.xml"'
        },
    )
