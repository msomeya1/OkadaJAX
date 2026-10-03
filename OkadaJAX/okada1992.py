import jax.numpy as jnp
from .utils import (_UA0, _UB0, _UC0, _UA, _UB, _UC, PI2, fault_geometry, medium_constants,
                    point_geometry,
                    _blank_where, _dummy_where, _length_scale, _rel_eps, _snap)





def DC3D0(ALPHA, X, Y, Z, DEPTH, DIP, POT1, POT2, POT3, POT4, 
          compute_strain=True, is_degree=True):
    """
    Displacement and strain at depth due to buried point source 
    in a semi-infinite medium.

    Parameters
    ----------
    ALPHA : float or JAX array
        Medium constant. (lambda+myu)/(lambda+2*myu)
    X, Y, Z : JAX array
        Coordinate of observing point.
    DEPTH : float or JAX array
        Source depth.
    DIP : JAX array
        Dip-angle.
    POT1, POT2, POT3, POT4 : float or JAX array
        Strike-, dip-, tensile- and inflate-potency.
        potency = (moment of double-couple)/myu for POT1,2
        potency = (intensity of isotropic part)/lambda for POT3
        potency = (intensity of linear dipole)/myu for POT4
    compute_strain : bool, default True
        Option to calculate the spatial derivative of the displacement. 
        New in the JAX implementation.
    is_degree : bool, default True
        Flag if `DIP` is in degree or not (= in radian). 
        New in the JAX implementation.

    Returns
    -------
    U : list of JAX array
        If `compute_strain` is `True`, 
        U is a list of 3 displacements and 9 spatial derivatives:
        [UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ]
        If `False`, U is a list of 3 displacements only:
        [UX, UY, UZ]

        UX, UY, UZ : JAX array
            Displacement. unit = (unit of potency) / (unit of X,Y,Z,DEPTH)**2
        UXX, UYX, UZX : JAX array
            X-derivative. unit = (unit of potency) / (unit of X,Y,Z,DEPTH)**3
        UXY, UYY, UZY : JAX array
            Y-derivative. unit = (unit of potency) / (unit of X,Y,Z,DEPTH)**3
        UXZ, UYZ, UZZ : JAX array
            Z-derivative. unit = (unit of potency) / (unit of X,Y,Z,DEPTH)**3

    IRET : JAX array (int)
        Return code.
        IRET=0 means normal,
        IRET=1 means singular,
        IRET=2 means positive z was given.


    Notes
    -----
    Original FORTRAN code was written by Y.Okada in Sep.1991, 
    revised in Nov.1991, May.2002.
    JAX implementation by M.Someya, 2025.
    """
    

    # Initialization
    N_variable = 12 if compute_strain else 3
    U = [jnp.zeros_like(X) for _ in range(N_variable)]
    IRET = jnp.zeros_like(X, dtype=jnp.int32)

    # Flag the unusable stations up front and hand them a dummy geometry, so
    # every intermediate stays finite; their output is zeroed at the end.
    # Masking only the output would leave the gradient NaN -- see `_dummy_where`.
    IRET = jnp.where(Z > 0.0, 2, IRET)                                     # above the surface
    # station at the source.  point_geometry snaps relative to R itself, so R == 0
    # exactly when all three coordinates are zero.
    IRET = jnp.where(X**2 + Y**2 + (DEPTH + Z)**2 == 0.0, 1, IRET)
    unusable = IRET != 0
    X, Y = _dummy_where(unusable, X, Y)

    C0 = medium_constants(ALPHA, DIP, is_degree)


    # REAL-SOURCE CONTRIBUTION
    DD = DEPTH + Z
    C1 = point_geometry(X, Y, DD, C0)
        
    DUA = _UA0(X, Y, DD, POT1, POT2, POT3, POT4, C0, C1, compute_strain)
    if compute_strain:
        for I in range(9):
            U[I] = U[I] - DUA[I]
        for I in range(3):
            U[I+9] = U[I+9] + DUA[I+9]
    else:
        for I in range(3):
            U[I] = U[I] - DUA[I]


    # IMAGE-SOURCE CONTRIBUTION
    DD = DEPTH - Z
    C1 = point_geometry(X, Y, DD, C0)

    DUA = _UA0(X, Y, DD, POT1, POT2, POT3, POT4, C0, C1, compute_strain)
    DUB = _UB0(X, Y, DD, Z, POT1, POT2, POT3, POT4, C0, C1, compute_strain)
    DUC = _UC0(X, Y, DD, Z, POT1, POT2, POT3, POT4, C0, C1, compute_strain)

    for I in range (N_variable):
        DU = DUA[I] + DUB[I] + Z * DUC[I]
        if I >= 9:
            DU = DU + DUC[I-9]
        U[I] = U[I] + DU

    U = _blank_where(unusable, U)

    return U, IRET
    




def DC3D(ALPHA, X, Y, Z, DEPTH, DIP, AL1, AL2, AW1, AW2, DISL1, DISL2, DISL3, 
         compute_strain=True, is_degree=True):
    """
    Displacement and strain at depth due to buried finite fault 
    in a semi-infinite medium.

    Parameters
    ----------
    ALPHA : float or JAX array
        Medium constant. (lambda+myu)/(lambda+2*myu)
    X, Y, Z : JAX array
        Coordinate of observing point.
    DEPTH : float or JAX array
        Depth of reference point.
    DIP : JAX array
        Dip-angle.
    AL1, AL2 : float or JAX array
        Fault length range.
    AW1, AW2 : float or JAX array
        Fault width range.
    DISL1, DISL2, DISL3 : float or JAX array
        Strike-, dip-, tensile-dislocations.
    compute_strain : bool, default True
        Option to calculate the spatial derivative of the displacement.
        New in the JAX implementation.
    is_degree : bool, default True
        Flag if `DIP` is in degree or not (= in radian). 
        New in the JAX implementation.

    Returns
    -------
    U : list of JAX array
        If `compute_strain` is `True`, 
        U is a list of 3 displacements and 9 spatial derivatives:
        [UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ]
        If `False`, U is a list of 3 displacements only:
        [UX, UY, UZ]

        UX, UY, UZ : JAX array
            Displacement. unit = (unit of dislocation)
        UXX, UYX, UZX : JAX array
            X-derivative. unit = (dislocation) / (unit of X,Y,Z,DEPTH,AL,AW)
        UXY, UYY, UZY : JAX array
            Y-derivative. unit = (dislocation) / (unit of X,Y,Z,DEPTH,AL,AW)
        UXZ, UYZ, UZZ : JAX array
            Z-derivative. unit = (dislocation) / (unit of X,Y,Z,DEPTH,AL,AW)
            
    IRET : JAX array (whose dtype is int)
        Return code.
        IRET=0 means normal,
        IRET=1 means singular,
        IRET=2 means positive z was given.


    Notes
    -----
    Original FORTRAN code was written by Y.Okada in Sep.1991, 
    revised in Nov.1991, Apr.1992, May.1993, Jul.1993, May.2002.
    JAX implementation by M.Someya, 2026.
    """


    # Initialization
    N_variable = 12 if compute_strain else 3
    U = [jnp.zeros_like(X) for _ in range(N_variable)]
    DU = [None] * N_variable
    XI = [jnp.zeros_like(X) for _ in range(2)]
    ET = [jnp.zeros_like(X) for _ in range(2)]
    KXI = [jnp.zeros_like(X, dtype=jnp.int32) for _ in range(2)]
    KET = [jnp.zeros_like(X, dtype=jnp.int32) for _ in range(2)]
    IRET = jnp.zeros_like(X, dtype=jnp.int32)

    IRET = jnp.where(
        Z > 0.0,
        2,
        IRET
    )
    above_surface = Z > 0.0

    C0 = medium_constants(ALPHA, DIP, is_degree)
    SD, CD = C0.SD, C0.CD  


    # "On the fault edge" is measured against the size of the fault, so the
    # test means the same thing in metres and in kilometres.
    fault_scale = _length_scale(AL2 - AL1, AW2 - AW1, reference=X)

    XI[0] = _snap(X - AL1, fault_scale)
    XI[1] = _snap(X - AL2, fault_scale)

    
    # REAL-SOURCE CONTRIBUTION
    D = DEPTH + Z
    P = Y * CD + D * SD
    Q = _snap(Y * SD - D * CD, fault_scale)
    ET[0] = _snap(P - AW1, fault_scale)
    ET[1] = _snap(P - AW2, fault_scale)


    # REJECT SINGULAR CASE
    # ON FAULT EDGE
    mask1 = jnp.logical_and(Q == 0.0, jnp.logical_or( 
        jnp.logical_and(XI[0] * XI[1] <= 0.0, ET[0] * ET[1] == 0.0),
        jnp.logical_and(ET[0] * ET[1] <= 0.0, XI[0] * XI[1] == 0.0)
    ))
    IRET = jnp.where(
        mask1,
        1,
        IRET
    )
    unusable_real = jnp.logical_or(above_surface, mask1)

    
    ## ON NEGATIVE EXTENSION OF FAULT EDGE
    R12 = jnp.sqrt(XI[0]**2 + ET[1]**2 + Q**2)
    R21 = jnp.sqrt(XI[1]**2 + ET[0]**2 + Q**2)
    R22 = jnp.sqrt(XI[1]**2 + ET[1]**2 + Q**2)

    KXI[0] = jnp.where(
        jnp.logical_and(XI[0] < 0.0, R21 + XI[1] < _rel_eps(X) * R21),
        1,
        0
    )
    KXI[1] = jnp.where(
        jnp.logical_and(XI[0] < 0.0, R22 + XI[1] < _rel_eps(X) * R22),
        1,
        0
    )
    KET[0] = jnp.where(
        jnp.logical_and(ET[0] < 0.0, R12 + ET[1] < _rel_eps(X) * R12),
        1,
        0
    )
    KET[1] = jnp.where(
        jnp.logical_and(ET[0] < 0.0, R22 + ET[1] < _rel_eps(X) * R22),
        1,
        0
    )
    

    for K in range(2):
        for J in range(2):
            # Dummy geometry for the unusable stations; see `_dummy_where`.
            XI_s, ET_s, Q_s = _dummy_where(unusable_real, XI[J], ET[K], Q)
            KXI_s, KET_s = [jnp.where(unusable_real, 0, k) for k in (KXI[K], KET[J])]
            C2 = fault_geometry(XI_s, ET_s, Q_s, SD, CD, KXI_s, KET_s)
            DUA = _UA(XI_s, ET_s, Q_s, DISL1, DISL2, DISL3, C0, C2, compute_strain)

            if compute_strain:
                for I in range(0, 10, 3):
                    DU[I]   = -DUA[I]
                    DU[I+1] = -DUA[I+1] * CD + DUA[I+2] * SD
                    DU[I+2] = -DUA[I+1] * SD - DUA[I+2] * CD
                    if I >= 9:
                        DU[I]   = -DU[I]
                        DU[I+1] = -DU[I+1]
                        DU[I+2] = -DU[I+2]
            else:
                DU[0] = -DUA[0]
                DU[1] = -DUA[1] * CD + DUA[2] * SD
                DU[2] = -DUA[1] * SD - DUA[2] * CD


            for I in range(N_variable):
                if (J + K == 1):
                    U[I] = U[I] - DU[I]
                else:
                    U[I] = U[I] + DU[I]



    # IMAGE-SOURCE CONTRIBUTION
    D = DEPTH - Z
    P = Y * CD + D * SD
    Q = _snap(Y * SD - D * CD, fault_scale)
    ET[0] = _snap(P - AW1, fault_scale)
    ET[1] = _snap(P - AW2, fault_scale)


    # REJECT SINGULAR CASE
    # ON FAULT EDGE
    mask2 = jnp.logical_and(Q == 0.0, jnp.logical_or( 
        jnp.logical_and(XI[0] * XI[1] <= 0.0, ET[0] * ET[1] == 0.0),
        jnp.logical_and(ET[0] * ET[1] <= 0.0, XI[0] * XI[1] == 0.0)
    ))
    IRET = jnp.where(
        mask2,
        1,
        IRET
    )
    unusable_image = jnp.logical_or(above_surface, mask2)
    
    
    ## ON NEGATIVE EXTENSION OF FAULT EDGE
    R12 = jnp.sqrt(XI[0]**2 + ET[1]**2 + Q**2)
    R21 = jnp.sqrt(XI[1]**2 + ET[0]**2 + Q**2)
    R22 = jnp.sqrt(XI[1]**2 + ET[1]**2 + Q**2)
    
    KXI[0] = jnp.where(
        jnp.logical_and(XI[0] < 0.0, R21 + XI[1] < _rel_eps(X) * R21),
        1,
        0
    )
    KXI[1] = jnp.where(
        jnp.logical_and(XI[0] < 0.0, R22 + XI[1] < _rel_eps(X) * R22),
        1,
        0
    )
    KET[0] = jnp.where(
        jnp.logical_and(ET[0] < 0.0, R12 + ET[1] < _rel_eps(X) * R12),
        1,
        0
    )
    KET[1] = jnp.where(
        jnp.logical_and(ET[0] < 0.0, R22 + ET[1] < _rel_eps(X) * R22),
        1,
        0
    )
    


    for K in range(2):
        for J in range(2):
            XI_s, ET_s, Q_s = _dummy_where(unusable_image, XI[J], ET[K], Q)
            KXI_s, KET_s = [jnp.where(unusable_image, 0, k) for k in (KXI[K], KET[J])]
            C2 = fault_geometry(XI_s, ET_s, Q_s, SD, CD, KXI_s, KET_s)
            DUA = _UA(XI_s, ET_s, Q_s, DISL1, DISL2, DISL3, C0, C2, compute_strain)
            DUB = _UB(XI_s, ET_s, Q_s, DISL1, DISL2, DISL3, C0, C2, compute_strain)
            DUC = _UC(XI_s, ET_s, Q_s, Z, DISL1, DISL2, DISL3, C0, C2, compute_strain)

            if compute_strain:
                for I in range(0, 10, 3):
                    DU[I]   = DUA[I] + DUB[I] + Z * DUC[I]
                    DU[I+1] = (DUA[I+1] + DUB[I+1] + Z * DUC[I+1]) * CD - (DUA[I+2] + DUB[I+2] + Z * DUC[I+2]) * SD
                    DU[I+2] = (DUA[I+1] + DUB[I+1] - Z * DUC[I+1]) * SD + (DUA[I+2] + DUB[I+2] - Z * DUC[I+2]) * CD
                    if I >= 9:
                        DU[ 9] = DU[ 9] + DUC[0]
                        DU[10] = DU[10] + DUC[1] * CD - DUC[2] * SD
                        DU[11] = DU[11] - DUC[1] * SD - DUC[2] * CD
            else:
                DU[0] =  DUA[0] + DUB[0] + Z * DUC[0]
                DU[1] = (DUA[1] + DUB[1] + Z * DUC[1]) * CD - (DUA[2] + DUB[2] + Z * DUC[2]) * SD
                DU[2] = (DUA[1] + DUB[1] - Z * DUC[1]) * SD + (DUA[2] + DUB[2] - Z * DUC[2]) * CD


            for I in range(N_variable):
                if (J + K == 1):
                    U[I] = U[I] - DU[I]
                else:
                    U[I] = U[I] + DU[I]
                    


    U = _blank_where(IRET != 0, U)

    return U, IRET
